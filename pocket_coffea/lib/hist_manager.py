import hist
import numpy as np
import awkward as ak
from collections import defaultdict
from coffea.analysis_tools import PackedSelection
from typing import List, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import logging


@dataclass
class Axis:
    field: str  # variable to plot
    label: str  # human readable label for the axis
    bins: int = None
    start: float = None
    stop: float = None
    coll: str = "events"  # Collection or events or metadata or custom
    name: str = None  # Identifier of the axis: By default is built as coll.field, if not provided
    pos: int = None  # index in the collection to plot. If None plot all the objects on the same histogram
    type: str = "regular"  # regular/variable/integer/intcat/strcat
    transform: str = None
    lim: Tuple[float] = (0, 0)
    underflow: bool = True
    overflow: bool = True
    growth: bool = False


@dataclass
class HistConf:
    axes: List[Axis]
    storage: str = "weight"
    autofill: bool = True  # Handle the filling automatically
    variations: bool = True
    only_variations: List[str] = None
    exclude_samples: List[str] = None
    only_samples: List[str] = None
    exclude_categories: List[str] = None
    only_categories: List[str] = None
    no_weights: bool = False  # Do not fill the weights
    metadata_hist: bool = False  # Non-event variables, for processing metadata
    hist_obj = None
    collapse_2D_masks = False  # if 2D masks are applied on the events
    # and the data_ndim=1, when collapse_2D_mask=True the OR
    # of the masks on the axis=2 is performed to get the mask
    # on axis=1, otherwise an exception is raised
    collapse_2D_masks_mode = "OR"  # Use OR or AND to collapse 2D masks for data_ndim=1 if collapse_2D_masks == True
    combine_subsamples = True
    # When True, disable custom fast accumulation and always fill via the public hist API
    # This is useful to compare the "custom fill" vs "hist fill API" behavior/performance per-histogram
    fill_via_hist_api_only: bool = True

    def serialize(self):
        out = {**self.__dict__}
        out["axes"] = []
        for a in self.axes:
            ax_dict = {}
            for k,v in a.__dict__.items():
                if k !="transform":
                    ax_dict[k] = v
            out["axes"].append(ax_dict)
        return out


def get_hist_axis_from_config(ax: Axis):
    if ax.name is None:
        ax.name = f"{ax.coll}.{ax.field}"
    if ax.type == "regular" and isinstance(ax.bins, list):
        ax.type = "variable"
    if ax.type == "regular":
        return hist.axis.Regular(
            name=ax.name,
            bins=ax.bins,
            start=ax.start,
            stop=ax.stop,
            label=ax.label,
            transform=ax.transform,
            overflow=ax.overflow,
            underflow=ax.underflow,
            growth=ax.growth,
        )
    elif ax.type == "variable":
        if not isinstance(ax.bins, list):
            raise ValueError(
                "A list of bins edges is needed as 'bins' parameters for a type='variable' axis"
            )
        return hist.axis.Variable(
            ax.bins,
            name=ax.name,
            label=ax.label,
            overflow=ax.overflow,
            underflow=ax.underflow,
            growth=ax.growth,
        )
    elif ax.type == "int":
        return hist.axis.Integer(
            name=ax.name,
            start=ax.start,
            stop=ax.stop,
            label=ax.label,
            overflow=ax.overflow,
            underflow=ax.underflow,
            growth=ax.growth,
        )
    elif ax.type == "intcat":
        return hist.axis.IntCategory(
            ax.bins,
            name=ax.name,
            label=ax.label,
            overflow=ax.overflow,
            underflow=ax.underflow,
            growth=ax.growth,
        )
    elif ax.type == "strcat":
        return hist.axis.StrCategory(
            ax.bins, name=ax.name, label=ax.label, growth=ax.growth
        )


def weights_cache(fun):
    '''
    Function decorator to cache the weights calculation when they are ndim=1 on data_structure of ndim=1.
    The weight is cached by (category, subsample, variation)
    '''
    def inner(self, category, subsample, variation, weight, mask, data_structure):
        if mask.ndim == 2:
            # Do not cache
            return fun(self, weight, mask, data_structure)
        #Cache only in the "by event" weight, which does not need to be
        #broadcasted on the data dimension.
        elif mask.ndim == 1 and (
            (data_structure is None) or (data_structure.ndim == 1)
        ):
            name = (category, subsample, variation)
            if name not in self._weights_cache:
                self._weights_cache[name] = fun(self, weight, mask, data_structure)

            return self._weights_cache[name]
        else:
            # if the mask is 2d, do not cache
            return fun(self, weight, mask, data_structure)
    return inner

class HistManager:
    def __init__(
        self,
        hist_config,
        year,
        sample,
        subsamples,
        categories_config,
        variations_config,
        weights_manager,
        processor_params,
        custom_axes=None,
        isMC=True,
        combine_subsamples=True,
    ):
        self.processor_params = processor_params
        self.isMC = isMC
        self.year = year
        self.subsamples = subsamples
        self.weights_manager = weights_manager
        self.histograms = defaultdict(dict)
        self.variations_config = variations_config
        self.categories_config = categories_config
        self.available_categories = set(self.categories_config.keys())
        self.available_weights_variations = ["nominal"]
        self.available_shape_variations = []
        # This dictionary is used to store the weights in some cases for performance reaso
        self._weights_cache = {}
        # Cache axes metadata per (subsample, histogram name)
        self._axes_cache = {}
        # Cache flattened storage views (values/variances) per histogram object id
        self._storage_views_cache = {}
        # Combine subsamples into a single histogram with a 'sample' axis
        # This reduces the number of histogram objects and allows faster traversal
        self.combine_subsamples = combine_subsamples
        # Map wildcard systematics to expanded subvariations
        self.wildcard_variations = {}

        def _resolve_storage(storage):
            """Map string storage names to hist.storage instances; pass through existing storage objects."""
            if isinstance(storage, str):
                mapping = {
                    "weight": hist.storage.Weight(),
                    "double": hist.storage.Double(),
                    "int64": hist.storage.Int64(),
                }
                if storage not in mapping:
                    raise ValueError(f"Unknown storage spec: {storage}")
                return mapping[storage]
            return storage
        # expose as method for later use
        self._resolve_storage = _resolve_storage

        # We take the variations config and we build the available variations
        # for each category and for the whole sample (if MC)
        # asking to the WeightsManager the available variations for the current specific chunk and metadata.
        self.available_weights_variations_bycat = defaultdict(list)
        self.available_shape_variations_bycat = defaultdict(list)
            
        if self.isMC:
            # Weights variations
            for cat, weights in self.variations_config["weights"].items():
                self.available_weights_variations_bycat[cat].append("nominal")
                for weight in weights:
                    # Ask the WeightsManager the available variations
                    vars = self.weights_manager.get_available_modifiers_byweight(weight)
                    self.available_weights_variations += vars
                    self.available_weights_variations_bycat[cat] += vars
            
            # Shape variations
            for cat, vars in self.variations_config["shape"].items():
                for var in vars:
                    # Check if the variation is a wildcard and the systematic requested has subvariations
                    # defined in the parameters
                    if (
                        var
                        in self.processor_params.systematic_variations.shape_variations
                    ):
                        for (
                            subvariation
                        ) in self.processor_params.systematic_variations.shape_variations[
                            var
                        ][
                            self.year
                        ]:
                            self.wildcard_variations[var] = f"{var}_{subvariation}"
                            self.available_weights_variations += [
                                f"{var}_{subvariation}Up",
                                f"{var}_{subvariation}Down",
                            ]
                            self.available_weights_variations_bycat[cat] += [
                                f"{var}_{subvariation}Up",
                                f"{var}_{subvariation}Down",
                            ]
                    else:
                        vv = [f"{var}Up", f"{var}Down"]
                        self.available_shape_variations += vv
                        self.available_shape_variations_bycat[cat] += vv
        else:  # DATA
            # Add a "weight_variation" nominal for data in each category
            for cat in self.categories_config.keys():
                self.available_weights_variations += ["nominal"]
                self.available_weights_variations_bycat[cat].append("nominal")
                
            
        
        # Reduce to set over all the categories
        self.available_weights_variations = set(self.available_weights_variations)
        self.available_shape_variations = set(self.available_shape_variations)
        # Prepare the variations Axes summing all the required variations
        # The variation config is organized as the weights one, by sample and by category
        for name, hcfg in deepcopy(hist_config).items():
            # Check if the histogram is active for the current sample
            # We only check for the parent sample, not for subsamples
            if hcfg.only_samples is not None:
                if sample not in hcfg.only_samples:
                    continue
            elif hcfg.exclude_samples is not None:
                if sample in hcfg.exclude_samples:
                    continue
            # Now we handle the selection of the categories
            cats = []
            for c in self.available_categories:
                if hcfg.only_categories is not None:
                    if c in hcfg.only_categories:
                        cats.append(c)
                elif hcfg.exclude_categories is not None:
                    if c not in hcfg.exclude_categories:
                        cats.append(c)
                else:
                    cats.append(c)
            # Update the histConf to save the only category
            hcfg.only_categories = list(sorted(cats))
            # Decide meta-axes encoding based on the first dense axis type
            dense_axis_types = [ax.type if ax.type != "variable" else "int" for ax in hcfg.axes]
            main_axis_type = dense_axis_types[0] if len(dense_axis_types) > 0 else "int"
            meta_axes_mode = "regular" if main_axis_type == "regular" else "int"
            # Build label->code mappings for meta axes
            hcfg._meta_axes_mode = meta_axes_mode
            hcfg._cat_labels = list(hcfg.only_categories)
            hcfg._cat_label_to_code = {c: i for i, c in enumerate(hcfg._cat_labels)}
            # Helper to create a numeric meta axis
            def _make_meta_axis(mode, n_bins, name, label):
                if mode == "int":
                    return hist.axis.Integer(0, n_bins, name=name, label=label, overflow=False, underflow=False, growth=False)
                else:
                    # Regular axis with unit-width bins centered on integer codes
                    return hist.axis.Regular(n_bins, -0.5, n_bins - 0.5, name=name, label=label, overflow=False, underflow=False, growth=False)
            # Create categories axis (numeric-encoded)
            cat_ax = _make_meta_axis(meta_axes_mode, len(hcfg._cat_labels), name="cat", label="Category")

            # Variation axes
            if hcfg.variations:
                # Get all the variation
                allvariat = self.available_weights_variations.union(self.available_shape_variations)
                
                if hcfg.only_variations is not None:
                    # expand wild card and Up/Down
                    only_variations = []
                    for var in hcfg.only_variations:
                        if var in self.wildcard_variations:
                            only_variations += [
                                f"{self.wildcard_variations[var]}Up",
                                f"{self.wildcard_variations[var]}Down",
                            ]
                        else:
                            only_variations += [
                                f"{var}Up",
                                f"{var}Down",
                            ]
                    # filtering the variation list with the available ones
                    allvariat = set(
                        filter(lambda v: v in only_variations or v == "nominal", allvariat)
                    )
                # sorted is needed to assure to have always the same order for all chunks
                hcfg.only_variations = list(sorted(set(allvariat)))
            else:
                hcfg.only_variations = ["nominal"]
            # Defining the variation axis (numeric-encoded)
            hcfg._variation_labels = list(hcfg.only_variations)
            hcfg._variation_label_to_code = {v: i for i, v in enumerate(hcfg._variation_labels)}
            var_ax = _make_meta_axis(meta_axes_mode, len(hcfg._variation_labels), name="variation", label="Variation")

            # Axis in the configuration + custom axes
            all_axes = [cat_ax]
            if self.isMC:
                all_axes.append(var_ax)
            # Add sample axis when combining subsamples
            if self.combine_subsamples and len(self.subsamples) > 0:
                hcfg._subsample_labels = list(self.subsamples)
                hcfg._subsample_label_to_code = {s: i for i, s in enumerate(hcfg._subsample_labels)}
                subs_ax = _make_meta_axis(meta_axes_mode, len(hcfg._subsample_labels), name="subsample", label="Subsample")
                all_axes.append(subs_ax)
            # the custom axis get included in the hcfg for future use
            if custom_axes:
                hcfg.axes = custom_axes + hcfg.axes
            # Cache per-axis arithmetic binning metadata for dense axes
            hcfg._axis_meta = {}
            for ax in hcfg.axes:
                if ax.type in ["regular", "int"]:
                    hcfg._axis_meta[ax.name or f"{ax.coll}.{ax.field}"] = {
                        "type": ax.type,
                        "start": ax.start,
                        "stop": ax.stop,
                        "bins": ax.bins,
                        "transform": ax.transform,
                        "growth": bool(getattr(ax, "growth", False)),
                        "width": None if not (ax.type == "regular" and ax.start is not None and ax.stop is not None and ax.bins)
                                  else (ax.stop - ax.start) / float(ax.bins),
                    }
            # Precompute capability flags for fast paths
            has_cat_dense = any(ax.type in ["strcat", "intcat"] for ax in hcfg.axes)
            has_growth_dense = any(bool(getattr(ax, "growth", False)) for ax in hcfg.axes)
            only_numeric_dense = all(ax.type in ["regular", "variable", "int"] for ax in hcfg.axes)
            event_level_by_config = all(
                (ax.coll in ["events", "metadata", "custom"]) or (ax.pos is not None)
                for ax in hcfg.axes
            )
            hcfg._slice_accumulate_ok = only_numeric_dense and (not has_cat_dense) and (not has_growth_dense)
            # Then we add those axes to the full list
            for ax in hcfg.axes:
                all_axes.append(get_hist_axis_from_config(ax))
            if self.combine_subsamples and len(self.subsamples) > 0:
                # Create a single histogram object shared across subsamples
                hcfg_shared = deepcopy(hcfg)
                hcfg_shared.hist_obj = hist.Hist(
                    *all_axes, storage=self._resolve_storage(hcfg.storage), name="Counts"
                )
                for subsample in self.subsamples:
                    self.histograms[subsample][name] = hcfg_shared
            else:
                # Creating an histogram object for each subsample
                for subsample in self.subsamples:
                    hcfg_subs = deepcopy(hcfg)
                    # Build the histogram object with the additional axes
                    hcfg_subs.hist_obj = hist.Hist(
                        *all_axes, storage=self._resolve_storage(hcfg.storage), name="Counts"
                    )
                    # Save the hist in the configuration and store the full config object
                    self.histograms[subsample][name] = hcfg_subs

    def get_histograms(self, subsample):
        # Exclude by default metadata histo
        out = {}
        for key, h in self.histograms[subsample].items():
            if h.metadata_hist:
                continue
            H = self._decode_meta_axes(h, h.hist_obj)
            # If subsamples were combined, project to the requested subsample to hide the 'sample' axis downstream
            if self.combine_subsamples and len(self.subsamples) > 0:
                try:
                    _ = H.axes["subsample"]
                    H = H[{"subsample": subsample}]
                except Exception:
                    pass
            out[key] = H
        return out

    def get_metadata_histograms(self, subsample):
        out = {}
        for key, h in self.histograms[subsample].items():
            if not h.metadata_hist:
                continue
            H = self._decode_meta_axes(h, h.hist_obj)
            if self.combine_subsamples and len(self.subsamples) > 0:
                try:
                    _ = H.axes["subsample"]
                    H = H[{"subsample": subsample}]
                except Exception:
                    pass
            out[key] = H
        return out

    def get_histogram(self, subsample, name):
        hcfg = self.histograms[subsample].get(name, None)
        if hcfg is None:
            return None
        hcfg_copy = deepcopy(hcfg)
        hcfg_copy.hist_obj = self._decode_meta_axes(hcfg, hcfg.hist_obj)
        return hcfg_copy

    def __prefetch_weights(self, category, shape_variation):
        '''
        Prefetch the weights for the category and the shape variation.
        - When processing the nominal shape variation we prefetch all the weights variations
        - When processing a shape variation we prefetch only the nominal weights
        '''
        weights = {}
        if shape_variation == "nominal":
            for variation in self.available_weights_variations_bycat[category]:
                if variation == "nominal":
                    weights["nominal"] = self.weights_manager.get_weight(category)
                else:
                    # Check if the variation is available in this category
                    weights[variation] = self.weights_manager.get_weight(
                        category, modifier=variation
                    )
        else:
            # Save only the nominal weights if a shape variation is being processed
            weights["nominal"] = self.weights_manager.get_weight(category)
        return weights

    # ---------- Small reusable helpers to simplify fill paths ----------
    def _build_dense_indices_from_vals(self, Hobj, histo, vals_np, axes_in_H=None):
        """
        Build per-dense-axis integer indices for values arrays using fast arithmetic
        when axis metadata allows it. Indices include flow bins according to axis traits
        and are clamped to valid ranges to avoid OOB when flows are disabled.
        """
        axes_in_H = axes_in_H or {ax.name: ax for ax in Hobj.axes}
        dense_idx_list = []
        for ax_conf in histo.axes:
            ax_name = ax_conf.name
            ax_obj = axes_in_H[ax_name]
            v = vals_np[ax_name]
            # Use cached axis meta if available
            meta = getattr(histo, "_axis_meta", {}).get(ax_name, None)
            # Fast arithmetic for Integer axes when safe
            if (meta and meta["type"] == "int") or (meta is None and ax_conf.type == "int"):
                start = (meta["start"] if meta else ax_conf.start)
                stop = (meta["stop"] if meta else ax_conf.stop)
                u = 1 if getattr(ax_obj.traits, 'underflow', False) else 0
                o = 1 if getattr(ax_obj.traits, 'overflow', False) else 0
                size = ax_obj.size
                vv = np.asarray(v, dtype=np.int64)
                raw = vv - start
                idx = np.where(vv < start, 0,
                               np.where(vv >= stop, (size + u) if o else (size + u - 1), raw + u))
                idx = np.clip(idx, 0, size + u + o - 1)
            # Fast arithmetic for Regular axes when safe (no transform/growth)
            elif (meta and meta["type"] == "regular" and (meta["transform"] is None) and (not meta["growth"])) \
                 or (meta is None and ax_conf.type == "regular" and (ax_conf.transform is None) and (not getattr(ax_conf, 'growth', False))):
                start = (meta["start"] if meta else ax_conf.start)
                stop = (meta["stop"] if meta else ax_conf.stop)
                bins = (meta["bins"] if meta else ax_conf.bins)
                u = 1 if getattr(ax_obj.traits, 'underflow', False) else 0
                o = 1 if getattr(ax_obj.traits, 'overflow', False) else 0
                size = ax_obj.size
                width = (meta["width"] if meta and meta["width"] is not None else (stop - start) / float(bins))
                vv = np.asarray(v, dtype=np.float64)
                raw = np.floor((vv - start) / width).astype(np.int64)
                idx = np.where(vv < start, 0,
                               np.where(vv >= stop, (size + u) if o else (size + u - 1), raw + u))
                idx = np.clip(idx, 0, size + u + o - 1)
            else:
                idx = ax_obj.index(v)
            dense_idx_list.append(idx)
        return dense_idx_list, axes_in_H

    def _axes_dims_with_flow(self, Hobj):
        dims = []
        for ax in Hobj.axes:
            size = ax.size + (1 if getattr(ax.traits, 'underflow', False) else 0) + (1 if getattr(ax.traits, 'overflow', False) else 0)
            dims.append(size)
        return dims

    def _linearize_indices(self, Hobj, indices_components):
        dims = self._axes_dims_with_flow(Hobj)
        return np.ravel_multi_index(tuple(indices_components), dims)

    def _linearize_indices_with_dims(self, dims, indices_components):
        return np.ravel_multi_index(tuple(indices_components), dims)

    def _get_storage_views(self, Hobj):
        """
        Return flattened writable views of histogram values (and variances if present),
        along with dtype and total size including flow. Uses ravel to avoid copies when possible.
        """
        key = id(Hobj)
        cached = self._storage_views_cache.get(key)
        if cached is not None:
            return cached
        vals_view = Hobj.values(flow=True)
        vals_flat = np.ravel(vals_view)
        if self._is_weight_storage(Hobj):
            vars_view = Hobj.variances(flow=True)
            vars_flat = np.ravel(vars_view)
        else:
            vars_flat = None
        out = (vals_flat, vars_flat, vals_view.dtype, vals_view.size)
        self._storage_views_cache[key] = out
        return out

    def _accumulate_into_hist(self, Hobj, lin_idx, values_flat, treat_as_counts=False):
        vals_flat, vars_flat, dtype, size = self._get_storage_views(Hobj)
        if vars_flat is not None:
            v = values_flat.astype(dtype, copy=False)
            bc_val = np.bincount(lin_idx, weights=v, minlength=size)
            vals_flat[:] += bc_val
            bc_var = np.bincount(lin_idx, weights=v * v, minlength=size)
            vars_flat[:] += bc_var
        else:
            if treat_as_counts:
                weights_arr = np.ones(lin_idx.shape[0], dtype=dtype)
            else:
                weights_arr = values_flat.astype(dtype, copy=False)
            bc = np.bincount(lin_idx, weights=weights_arr, minlength=size)
            vals_flat[:] += bc

    def _get_axes_cached(self, subsample, name, Hobj):
        key = id(Hobj)
        if key in self._axes_cache:
            return self._axes_cache[key]
        axes_in_H = {ax.name: ax for ax in Hobj.axes}
        dims = self._axes_dims_with_flow(Hobj)
        self._axes_cache[key] = (axes_in_H, dims)
        return self._axes_cache[key]

    def _is_weight_storage(self, Hobj):
        """Return True if histogram uses Weight storage (has variances)."""
        st = None
        try:
            st = Hobj._storage_type()
        except TypeError:
            st = getattr(Hobj, "_storage_type", None)
        return (st is hist.storage.Weight) or isinstance(st, hist.storage.Weight) or (st == hist.storage.Weight)

    def _get_storage_instance(self, H):
        """Return a storage instance matching H's storage type for constructing new hist objects."""
        st = None
        try:
            st = H._storage_type()
        except TypeError:
            st = getattr(H, "_storage_type", None)
        try:
            if (st is hist.storage.Weight) or isinstance(st, hist.storage.Weight) or (st == hist.storage.Weight):
                return hist.storage.Weight()
            if (st is hist.storage.Double) or isinstance(st, hist.storage.Double) or (st == hist.storage.Double):
                return hist.storage.Double()
            if (st is hist.storage.Int64) or isinstance(st, hist.storage.Int64) or (st == hist.storage.Int64):
                return hist.storage.Int64()
            # Try to instantiate as a callable class
            return st()
        except Exception:
            return hist.storage.Double()

    def _broadcast_custom_weight(self, cw, mask, data_structure):
        """
        Broadcast and mask a custom weight array to match the flattening rules for values.
        Returns a numpy array ready to be filtered by the non-None mask.
        """
        if data_structure is not None and getattr(data_structure, "ndim", None) == 2 and getattr(mask, "ndim", None) == 1:
            ds = data_structure
            if ak.sum(ak.is_none(ds, axis=-1)) > 0:
                ds = ak.fill_none(ds, 0.)
            return ak.to_numpy(ak.flatten(ds * (cw[mask])), allow_missing=False)
        elif getattr(mask, "ndim", None) == 2:
            return ak.to_numpy(ak.flatten((ak.ones_like(mask) * cw)[mask]), allow_missing=False)
        else:
            return ak.to_numpy(cw[mask], allow_missing=False)

    def fill_histograms(
        self,
        events,
        categories,
        shape_variation="nominal",
        subsamples=None,  # This is a dictionary with name:ak.Array(bool)
        custom_fields=None,
        custom_weight=None,  # it should be a dictionary {variable:weight}
    ):
        '''
        We loop on the configured histograms only
        Doing so the catergory, sample, variation selections are handled correctly (by the constructor).

        Custom_fields is a dict of additional array. The expected lenght of the first dimension is the number of
        events. The categories mask will be applied.
        '''

        # Cache category and subsample masks once per call (per variation)
        # and precompute combined masks to avoid recomputing for each histogram
        cat_masks_list = list(categories.get_masks())
        subs_masks_list = list(subsamples.get_masks())
        # Map category -> list of (subsample, combined_mask, nonempty)
        from collections import defaultdict as _dd
        cat_to_subs_masks = _dd(list)
        cat_has_any = {}
        for category, cat_mask in cat_masks_list:
            any_nonempty = False
            for subsample, subs_mask in subs_masks_list:
                comb_mask = cat_mask & subs_mask
                # Note: ak.sum is lazy; cast to int for truthiness
                nonempty = ak.sum(comb_mask) > 0
                cat_to_subs_masks[category].append((subsample, comb_mask, nonempty))
                any_nonempty = any_nonempty or bool(nonempty)
            cat_has_any[category] = any_nonempty
            
        # Cleaning the weights cache decorator between calls.
        self._weights_cache.clear()
        # Clear cached storage views per fill call
        self._storage_views_cache.clear()
        # Looping on the histograms to read the values only once
        # Then categories, subsamples and weights are applied and masked correctly

        # ASSUNTION, the histograms are the same for each subsample
        # we can take the configuration of the first subsample
        for name, histo in self.histograms[self.subsamples[0]].items():
            # logging.info(f"\thisto: {name}")
            if not histo.autofill:
                continue
            if histo.metadata_hist:
                continue  # TODO dedicated function for metadata histograms

            # Check if a shape variation is under processing and if this histogram has a variation axis
            has_var_axis = any(ax.name == "variation" for ax in histo.hist_obj.axes)
            if shape_variation != "nominal":
                if not has_var_axis:
                    continue
                if shape_variation not in histo._variation_labels:
                    continue

            # Get the filling axes --> without any masking.
            # The flattening has to be applied as the last step since the categories and subsamples
            # work at event level

            fill_categorical = {}
            fill_numeric = {}
            data_ndim = None

            for ax in histo.axes:
                # Checkout the collection type
                if ax.type in ["regular", "variable", "int"]:
                    if ax.coll == "events":
                        # These are event level information
                        data = events[ax.field]
                    elif ax.coll == "metadata":
                        data = events.metadata[ax.field]
                    elif ax.coll == "custom":
                        # taking the data from the custom_fields argument
                        # IT MUST be a per-event number, so we expect an array to mask
                        data = custom_fields[ax.field]
                    else:
                        if ax.coll not in events.fields:
                            raise ValueError(
                                f"Collection {ax.coll} not found in events!"
                            )
                        # General collections
                        if ax.pos == None:
                            data = events[ax.coll][ax.field]
                        elif ax.pos >= 0:
                            data = ak.pad_none(
                                events[ax.coll][ax.field], ax.pos + 1, axis=1
                            )[:, ax.pos]
                        else:
                            raise Exception(
                                f"Invalid position {ax.pos} requested for collection {ax.coll}"
                            )

                    # Flattening
                    if data_ndim == None:
                        data_ndim = data.ndim
                    elif data_ndim != data.ndim:
                        raise Exception(
                            f"Incompatible shapes for Axis {ax} of hist {histo}"
                        )
                    # If we have multidim data we need to flatten it
                    # but we will do it after the event masking of each category

                    # Filling the numerical axes
                    fill_numeric[ax.name] = data

                #### --> end of numeric axes
                # Categorical axes (not appling the mask)
                else:
                    if ax.coll == "metadata":
                        data = events.metadata[ax.field]
                        fill_categorical[ax.name] = data
                    elif ax.coll == "custom":
                        # taking the data from the custom_fields argument
                        data = custom_fields[ax.field]
                        fill_categorical[ax.name] = data
                    else:
                        raise NotImplementedError()

            # Now the variables have been read for all the events
            # We need now to iterate on categories and subsamples
            # Mask the events, the weights and then flatten and remove the None correctly
            # Restrict to only the categories this histogram actually uses
            active_categories = histo.only_categories if histo.only_categories is not None else list(self.available_categories)

            # Prefetch weights only for active categories (reduces overhead)
            weights = {}
            for category in active_categories:
                weights[category] = self.__prefetch_weights(category, shape_variation)
            # Note: custom_weight is handled inline where needed per path

            # decide once if fast category-slice accumulation is allowed
            # Disable if the per-hist flag requests using the public hist fill API only
            fast_slice_ok = (
                getattr(histo, "_slice_accumulate_ok", False)
                and data_ndim == 1
                and not getattr(histo, "fill_via_hist_api_only", False)
            )

            # No comparison mode: external profiler will be used. Nothing to set up here.
            for category, cat_mask in filter(lambda x: x[0] in active_categories, cat_masks_list):
                # Skip categories with no events across all subsamples for this variation
                if not cat_has_any.get(category, True):
                    continue
                # loop directly on subsamples (precomputed combined masks)
                for subsample, mask, nonempty in cat_to_subs_masks[category]:
                    # logging.info(f"\t\tcategory {category}, subsample {subsample}")
                    # Skip empty categories and subsamples
                    if not nonempty:
                        continue

                    # Check if the required data is dim=1, per event,
                    # and the mask is by collection.
                    # In this case the mask is reduced to per-event mask
                    # doing a logical OR only if explicitely allowed by the user
                    # WARNING!! POTENTIAL PROBLEMATIC BEHAVIOUR
                    # The user must be aware of the behavior.

                    if data_ndim == 1 and mask.ndim > 1:
                        if histo.collapse_2D_masks:
                            if histo.collapse_2D_masks_mode == "OR":
                                mask = ak.any(mask, axis=1)
                            elif histo.collapse_2D_masks_mode == "AND":
                                mask = ak.all(mask, axis=1)
                            else:
                                raise Exception(
                                    "You want to collapse the 2D masks on 1D data but the `collapse_2D_masks_mode` is not 'AND/OR'"
                                )

                        else:
                            raise Exception(
                                "+++++ BE AWARE! This is a possible mis-behavior! +++++\n"
                                + f"You are trying to fill the histogram {name} with data of dimention 1 (variable by event)"
                                + "and masking it with a mask with more than 1 dimension (e.g. mask on Jets)\n"
                                + "This means that you are either performing a cut on a collections (e.g Jets),"
                                + " or you are using subsamples with cuts on collections.\n"
                                + "\n As an example of a bad behaviour would be saving the pos=1 of a collection e.g. `JetGood.pt[1]`\n"
                                + "while also having a 2D cut on the `JetGood` collection --> this is not giving you the second jet passing the cut!\n"
                                + "In that case the 2nd JetGood.pt will be always plotted even if masked by the 2D cut: in fact "
                                + "the 2D masks would be collapsed to the event dimension. \n\n"
                                + "If you really wish to save the histogram with a single value for event (data dim=1)"
                                + "you can do so by configuring the histogram with `collapse_2D_masks=True\n"
                                + "The 2D masks will be collapsed on the event dimension (axis=1) doing an OR (default) or an AND\n"
                                + "You can configure this behaviour with `collapse_2D_masks_mode='OR'/'AND'` in the histo configuration."
                            )

                    # Mask the variables and flatten them
                    # save the isnotnone and datastructure
                    # to be able to broadcast the weight
                    has_none_mask = False
                    all_axes_isnotnone = None
                    has_data_structure = False
                    data_structure = None
                    fill_numeric_masked = {}
                    # loop on the cached numerical filling
                    for field, data in fill_numeric.items():
                        masked_data = data[mask]
                        # For each field we need to mask and flatten
                        if data_ndim > 1:
                            # We need to flatten and
                            # save the data structure for weights propagation
                            if not has_data_structure:
                                data_structure = ak.ones_like(masked_data)
                                has_data_structure = True
                            # flatten the data in one dimension
                            masked_data = ak.flatten(masked_data)

                        # check isnotnone AFTER the flattening
                        if not has_none_mask:  # this is the first axis analyzed
                            all_axes_isnotnone = ~ak.is_none(masked_data)
                            has_none_mask = True
                        else:
                            all_axes_isnotnone = all_axes_isnotnone & (
                                ~ak.is_none(masked_data)
                            )
                        # Save the data for the filling
                        fill_numeric_masked[field] = masked_data

                    # Now apply the isnone mask to all the numeric fields already masked
                    for key, value in fill_numeric_masked.items():
                        # we also convert it to numpy to speedup the hist filling
                        fill_numeric_masked[key] = ak.to_numpy(
                            value[all_axes_isnotnone], allow_missing=False
                        )
                    # Apply the same masking and conversion to categorical fields to ensure matching lengths
                    fill_categorical_masked = {}
                    for key, value in fill_categorical.items():
                        try:
                            v = value[mask]
                            if data_ndim and data_ndim > 1:
                                v = ak.flatten(v)
                            fill_categorical_masked[key] = ak.to_numpy(v[all_axes_isnotnone], allow_missing=False)
                        except Exception:
                            # Fallback: try to broadcast a scalar value if provided
                            try:
                                fill_categorical_masked[key] = np.full(
                                    fill_numeric_masked[next(iter(fill_numeric_masked))].shape[0],
                                    value,
                                )
                            except Exception:
                                # As a last resort, keep original (may raise during fill if mismatched)
                                fill_categorical_masked[key] = value

                    # Precompute common helpers for the current (category, subsample) slice
                    # - Reuse non-None mask for weight slicing
                    # - Cache axes mapping/dimensions and dense indices once per slice
                    nn_mask_np = ak.to_numpy(all_axes_isnotnone)
                    Hobj = self.histograms[subsample][name].hist_obj
                    axes_in_H, dims = self._get_axes_cached(subsample, name, Hobj)
                    dense_idx_list, _ = self._build_dense_indices_from_vals(Hobj, histo, fill_numeric_masked, axes_in_H=axes_in_H)
                    has_subs_axis = ("subsample" in axes_in_H)
                    subs_code = histo._subsample_label_to_code[subsample] if has_subs_axis else None
                    category_code = histo._cat_label_to_code[category]
                    # Infer number of rows to fill from any numeric field
                    n_rows_fill = fill_numeric_masked[next(iter(fill_numeric_masked))].shape[0]

                    # Preconvert numpy weights at the 'valid' mask level once for this category
                    np_weights_valid_by_var = {}
                    if shape_variation == "nominal":
                        # Include all weight variations used in this histogram
                        for variation in histo._variation_labels:
                            if variation in self.available_shape_variations:
                                continue
                            if variation in weights[category]:
                                np_weights_valid_by_var[variation] = ak.to_numpy(weights[category][variation][mask], allow_missing=False)
                        # Always include nominal
                        np_weights_valid_by_var["nominal"] = ak.to_numpy(weights[category]["nominal"][mask], allow_missing=False)
                    else:
                        np_weights_valid_by_var["nominal"] = ak.to_numpy(weights[category]["nominal"][mask], allow_missing=False)
                    # Ok, now we have all the numerical axes with
                    # data that has been masked, flattened
                    # removed the none value --> now we need weights for each variation
                    if not histo.no_weights and self.isMC:
                        if shape_variation == "nominal":
                            # Batch all weight variations into a single accumulation
                            var_labels = [v for v in histo._variation_labels if v not in self.available_shape_variations]
                            W_stack = []
                            var_codes = []
                            for variation in var_labels:
                                # Resolve variation weights (fallback to nominal if unavailable for this category)
                                if variation not in weights[category]:
                                    weight_varied = np_weights_valid_by_var["nominal"]
                                else:
                                    weight_varied = np_weights_valid_by_var.get(variation)
                                    if weight_varied is None:
                                        weight_varied = ak.to_numpy(weights[category][variation][mask], allow_missing=False)
                                # Broadcast and mask the weight (broadcast when needed)
                                if data_structure is not None and data_structure.ndim == 2 and mask.ndim == 1:
                                    ds = data_structure
                                    if ak.sum(ak.is_none(ds, axis=-1)) > 0:
                                        ds = ak.fill_none(ds, 0.)
                                    weight_varied = ak.to_numpy(ak.flatten(ds * (weight_varied)), allow_missing=False)
                                elif mask.ndim == 2:
                                    weight_varied = ak.to_numpy(ak.flatten((ak.ones_like(mask) * weight_varied)[mask]), allow_missing=False)
                                else:
                                    weight_varied = ak.to_numpy(weight_varied, allow_missing=False)
                                if custom_weight != None and name in custom_weight:
                                    cw = custom_weight[name][mask]
                                    if data_structure is not None and data_structure.ndim == 2 and mask.ndim == 1:
                                        ds = data_structure
                                        if ak.sum(ak.is_none(ds, axis=-1)) > 0:
                                            ds = ak.fill_none(ds, 0.)
                                        cw = ak.to_numpy(ak.flatten(ds * (cw)), allow_missing=False)
                                    elif mask.ndim == 2:
                                        cw = ak.to_numpy(ak.flatten((ak.ones_like(mask) * cw)[mask]), allow_missing=False)
                                    else:
                                        cw = ak.to_numpy(cw, allow_missing=False)
                                    weight_varied = weight_varied * cw
                                # Apply non-None mask and collect
                                W_stack.append(weight_varied[nn_mask_np])
                                var_codes.append(histo._variation_label_to_code[variation])
                            # Single accumulation for all variations if fast path is allowed
                            if fast_slice_ok and len(W_stack) > 0:
                                try:
                                    num_vars = len(W_stack)
                                    total = n_rows_fill * num_vars
                                    indices = [np.full(total, category_code, dtype=np.int32)]
                                    # Variation indices repeated per block of n_rows_fill
                                    indices.append(np.repeat(np.asarray(var_codes, dtype=np.int32), n_rows_fill))
                                    if has_subs_axis:
                                        indices.append(np.full(total, subs_code, dtype=np.int32))
                                    for idx_dense in dense_idx_list:
                                        indices.append(np.tile(np.asarray(idx_dense), num_vars))
                                    lin_idx = self._linearize_indices_with_dims(dims, indices)
                                    weights_concat = np.concatenate(W_stack)
                                    self._accumulate_into_hist(
                                        Hobj,
                                        lin_idx,
                                        weights_concat,
                                        treat_as_counts=False if self._is_weight_storage(Hobj) else True,
                                    )
                                    # Done with this (category, subsample) slice
                                    continue
                                except Exception:
                                    # fall back to standard per-variation fill below
                                    pass
                            # Fallback: single batched call via histogram API with tiled values
                            try:
                                num_vars = len(W_stack)
                                total = n_rows_fill * num_vars
                                # Build meta-axes arrays
                                cat_arr = np.full(total, category_code, dtype=np.int32)
                                var_arr = np.repeat(np.asarray(var_codes, dtype=np.int32), n_rows_fill)
                                kwargs_axes = {}
                                # Tile numeric and categorical values
                                for k, v in fill_numeric_masked.items():
                                    kwargs_axes[k] = np.tile(v, num_vars)
                                for k, v in fill_categorical_masked.items():
                                    try:
                                        kwargs_axes[k] = np.tile(v, num_vars)
                                    except Exception:
                                        kwargs_axes[k] = v
                                # Add meta axes
                                kwargs_axes["cat"] = cat_arr
                                if has_var_axis:
                                    kwargs_axes["variation"] = var_arr
                                if has_subs_axis:
                                    kwargs_axes["subsample"] = np.full(total, subs_code, dtype=np.int32)
                                # Concatenate weights
                                weights_concat = np.concatenate(W_stack)
                                # Single fill call
                                self.histograms[subsample][name].hist_obj.fill(
                                    weight=weights_concat,
                                    **kwargs_axes,
                                )
                            except Exception as e:
                                raise Exception(
                                    f"Cannot fill histogram (batched API): {name}, {histo} {e}"
                                )
                        else:
                            # Check if this shape variation is requested for this category
                            if shape_variation not in self.available_shape_variations_bycat[category]:
                                # it means that the variation is in the axes only
                                # because it is requested for another category.
                                # We cannot fill just with the nominal, because we are running the shape
                                # variation and the observable hist will be different, also if with nominal weights.
                                continue
                                
                            # Working on shape variation! only nominal weights
                            # (also using the cache which is cleaned for each shape variation
                            # at the beginning of the function)
                            # Use preconverted nominal
                            weights_nom = np_weights_valid_by_var["nominal"]
                            if data_structure is not None and data_structure.ndim == 2 and mask.ndim == 1:
                                ds = data_structure
                                if ak.sum(ak.is_none(ds, axis=-1)) > 0:
                                    ds = ak.fill_none(ds, 0.)
                                weights_nom = ak.to_numpy(ak.flatten(ds * (weights_nom)), allow_missing=False)
                            elif mask.ndim == 2:
                                weights_nom = ak.to_numpy(ak.flatten((ak.ones_like(mask) * weights_nom)[mask]), allow_missing=False)
                            else:
                                weights_nom = ak.to_numpy(weights_nom, allow_missing=False)
                            if custom_weight != None and name in custom_weight:
                                cw = custom_weight[name][mask]
                                if data_structure is not None and data_structure.ndim == 2 and mask.ndim == 1:
                                    ds = data_structure
                                    if ak.sum(ak.is_none(ds, axis=-1)) > 0:
                                        ds = ak.fill_none(ds, 0.)
                                    cw = ak.to_numpy(ak.flatten(ds * (cw)), allow_missing=False)
                                elif mask.ndim == 2:
                                    cw = ak.to_numpy(ak.flatten((ak.ones_like(mask) * cw)[mask]), allow_missing=False)
                                else:
                                    cw = ak.to_numpy(cw, allow_missing=False)
                                weights_nom = weights_nom * cw
                            # Then we apply the notnone mask
                            weights_nom = weights_nom[nn_mask_np]
                            # Optimized accumulation for shape variation as well
                            if fast_slice_ok:
                                try:
                                    indices = [np.full(n_rows_fill, category_code, dtype=np.int32)]
                                    if "variation" in axes_in_H:
                                        var_code = histo._variation_label_to_code[shape_variation]
                                        indices.append(np.full(n_rows_fill, var_code, dtype=np.int32))
                                    if has_subs_axis:
                                        indices.append(np.full(n_rows_fill, subs_code, dtype=np.int32))
                                    for idx_dense in dense_idx_list:
                                        indices.append(np.asarray(idx_dense))
                                    lin_idx = self._linearize_indices_with_dims(dims, indices)
                                    self._accumulate_into_hist(
                                        Hobj,
                                        lin_idx,
                                        weights_nom,
                                        treat_as_counts=False if self._is_weight_storage(Hobj) else True,
                                    )
                                    continue
                                except Exception:
                                    # fall back to standard fill below
                                    pass
                            # Fall back to standard fill
                            try:
                                if self.combine_subsamples and len(self.subsamples) > 0:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=category_code,
                                        variation=histo._variation_label_to_code[shape_variation],
                                        subsample=histo._subsample_label_to_code[subsample] if hasattr(histo, "_subsample_label_to_code") else None,
                                        weight=weights_nom,
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                                else:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=category_code,
                                        variation=histo._variation_label_to_code[shape_variation],
                                        weight=weights_nom,
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                            except Exception as e:
                                raise Exception(
                                    f"Cannot fill histogram: {name}, {histo} {e}"
                                )
                    ##################################################################################
                    elif not histo.no_weights and not self.isMC:   #DATA
                        # Broadcast unit weights (as before) and optionally multiply custom weight
                        weights_data = np.ones(n_rows_fill, dtype=np.float64)
                        if custom_weight is not None and name in custom_weight:
                            cw = self._broadcast_custom_weight(custom_weight[name], mask, data_structure)
                            weights_data = weights_data * cw
                        weights_data = weights_data[nn_mask_np]
                        try:
                            if self.combine_subsamples and len(self.subsamples) > 0:
                                self.histograms[subsample][name].hist_obj.fill(
                                    cat=histo._cat_label_to_code[category],
                                    subsample=histo._subsample_label_to_code[subsample] if hasattr(histo, "_subsample_label_to_code") else None,
                                    weight=weights_data,
                                    **{**fill_categorical, **fill_numeric_masked},
                                )
                            else:
                                self.histograms[subsample][name].hist_obj.fill(
                                    cat=histo._cat_label_to_code[category],
                                    weight=weights_data,
                                    **{**fill_categorical, **fill_numeric_masked},
                                )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram for Data: {name}, {histo} {e}"
                            )

                    ######################################################
                    elif (
                        histo.no_weights and self.isMC
                    ):  # NO Weights modifier for the histogram
                        try:
                            if self.combine_subsamples and len(self.subsamples) > 0:
                                # Pass variation only if the histogram has that axis
                                if has_var_axis:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=histo._cat_label_to_code[category],
                                        variation=histo._variation_label_to_code["nominal"],
                                        subsample=histo._subsample_label_to_code[subsample] if hasattr(histo, "_subsample_label_to_code") else None,
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                                else:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=histo._cat_label_to_code[category],
                                        subsample=histo._subsample_label_to_code[subsample] if hasattr(histo, "_subsample_label_to_code") else None,
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                            else:
                                if has_var_axis:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=histo._cat_label_to_code[category],
                                        variation=histo._variation_label_to_code["nominal"],
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                                else:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=histo._cat_label_to_code[category],
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram: {name}, {histo} {e}"
                            )

                    elif histo.no_weights and not self.isMC:
                        # Fill histograms for Data
                        try:
                            if self.combine_subsamples and len(self.subsamples) > 0:
                                self.histograms[subsample][name].hist_obj.fill(
                                    cat=histo._cat_label_to_code[category],
                                    subsample=histo._subsample_label_to_code[subsample] if hasattr(histo, "_subsample_label_to_code") else None,
                                    **{**fill_categorical, **fill_numeric_masked},
                                )
                            else:
                                self.histograms[subsample][name].hist_obj.fill(
                                    cat=histo._cat_label_to_code[category],
                                    **{**fill_categorical, **fill_numeric_masked},
                                )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram: {name}, {histo} {e}"
                            )
                    else:
                        raise Exception(
                            f"Cannot fill histogram: {name}, {histo}, not implemented combination of options"
                        )


        ###################
        # Utilities to handle the Weights cache

    @weights_cache
    def mask_and_broadcast_weight(self, weight, mask, data_structure):
        '''
        The function mask the weights and broadcast them to the correct dimension.
        The `data_structure` input is an array of 1-value with the structure of the data ALREADY masked.
        We need instead to mask the weight value and broadcast it.

        We need to handle different cases:
        - Mask dimension=1 (mask on events):
           If the data_structure.dim = 2 it means that we want to plot a collection
           - we mask the weights by events (data is already masked)
           - broadcast weight to the collection by multiplying to the datastructure (1-like array)
           - flatten the final weight
           If the data_structure.dim = 1:
           - We just mask the weight by event

        - Mask dimension=2 (mask on the collection)
          It means that we are masking the collection, not the events.
          - First we broadcast the weight to the structure of the mask
          - Then we apply the mask
          - Then we flatten the weight

        '''    
        if mask.ndim == 1 and not (data_structure is None) and data_structure.ndim == 2:
            # If the mask has dim =1 and the data dim =2
            # we need to mask the weight on dim=1, then to broadcast
            # on the data_structure -> then flatten
            allow_missing = False
            if ak.sum(ak.is_none(data_structure, axis=-1)) > 0:
                data_structure = ak.fill_none(data_structure, 0.)

            return ak.to_numpy(
                ak.flatten(data_structure * (weight[mask])), allow_missing=False
            )

        elif mask.ndim == 2:
            # First we broadcast the weight then we mask
            # if the mask is ndim==2 also the data is ndim==2.
            # The weights are broadcasted at collection level, then masked, then flattened.
            return ak.to_numpy(
                ak.flatten((ak.ones_like(mask) * weight)[mask]), allow_missing=False
            )
        else:
            return ak.to_numpy(weight[mask], allow_missing=False)

        
    def _decode_meta_axes(self, hcfg, H):
        """
        Convert numeric-encoded meta axes (cat/variation/subsample) back to StrCategory axes
        to preserve downstream compatibility. The operation preserves bin ordering and storage.
        """
        try:
            mode = getattr(hcfg, "_meta_axes_mode", None)
            if mode not in ("regular", "int"):
                return H
            # Build new meta axes with original labels
            new_axes = []
            names_existing = [ax.name for ax in H.axes]
            if "cat" in names_existing:
                new_axes.append(hist.axis.StrCategory(hcfg._cat_labels, name="cat", label="Category", growth=False))
            if "variation" in names_existing:
                new_axes.append(hist.axis.StrCategory(hcfg._variation_labels, name="variation", label="Variation", growth=False))
            if "subsample" in names_existing:
                new_axes.append(hist.axis.StrCategory(hcfg._subsample_labels, name="subsample", label="Subsample", growth=False))
            # Append the dense axes unchanged
            for ax in H.axes:
                if ax.name in ("cat", "variation", "subsample"):
                    continue
                new_axes.append(ax)
            Hnew = hist.Hist(*new_axes, storage=self._get_storage_instance(H), name=getattr(H, "name", "Counts"))
            # Copy content (including variances when present)
            try:
                Hnew_view = Hnew.view()
                H_view = H.view()
                Hnew_view[...] = H_view
            except Exception:
                # Fallback: copy values/variances explicitly
                Hnew.values()[...] = H.values()
                if self._is_weight_storage(H):
                    Hnew.variances()[...] = H.variances()
            return Hnew
        except Exception as e:
            raise Exception(f"Failed to decode meta axes: {e}")



