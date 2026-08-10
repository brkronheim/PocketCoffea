import hist
import awkward as ak
import numpy as np
from collections import defaultdict
from coffea.analysis_tools import PackedSelection
from typing import List, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import logging
from .weights.weights_manager import get_weights_by_cat_var, get_weights_by_cat_var_subsample


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

    def serialize(self):
        out = {**self.__dict__}
        out["axes"] = []
        for a in self.axes:
            ax_dict = {}
            for k,v in a.__dict__.items():
                if k !="transform":
                    ax_dict[k] = v
        return out


def get_hist_axis_from_config(ax: Axis):
    if ax.name == None:
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
        has_subsamples,
        subsamples,
        categories_config,
        variations_config,
        weights_manager,
        calibrators_manager,
        processor_params,
        custom_axes=None,
        isMC=True,
    ):
        self.processor_params = processor_params
        self.isMC = isMC
        self.year = year
        self.sample = sample
        self.has_subsamples = has_subsamples
        self.subsamples = subsamples
        self.weights_manager = weights_manager
        self.calibrators_manager = calibrators_manager
        self.histograms = defaultdict(dict)
        self.variations_config = variations_config
        self.categories_config = categories_config
        self.available_categories = set(self.categories_config.keys())
        self.available_weights_variations = ["nominal"]
        self.available_shape_variations = []
        # This dictionary is used to store the weights in some cases for performance reaso
        self._weights_cache = {}

        # We take the variations config and we build the available variations
        # for each category and for the whole sample (if MC)
        # asking to the WeightsManager the available variations for the current specific chunk and metadata.
        self.available_weights_variations_bycat = defaultdict(list)
        self.available_shape_variations_bycat = defaultdict(list)
        # Variations by subsabples
        if self.has_subsamples:
            self.available_weights_variations_bysubsample = {
                sub : [] for sub in self.subsamples
            }
            self.available_weights_variations_bysubsample_bycat = {
                sub : defaultdict(list) for sub in self.subsamples
            } 
            self.available_shape_variations_bysubsample = {
                sub : [] for sub in self.subsamples
            } 
            self.available_shape_variations_bysubsample_bycat = {
                sub : defaultdict(list) for sub in self.subsamples
            }
        else:
            self.available_weights_variations_bysubsample = None
            self.available_weights_variations_bysubsample_bycat = None
            self.available_shape_variations_bysubsample = None
            self.available_shape_variations_bysubsample_bycat = None
            
            
        if self.isMC:
            # Weights variations
            # This is checking only the full samples weights
            for cat, weights in self.variations_config["weights"].items():
                self.available_weights_variations_bycat[cat].append("nominal")
                for weight in weights:
                    # Ask the WeightsManager the available variations
                    vars = self.weights_manager.get_available_modifiers_byweight(weight)
                    self.available_weights_variations += vars
                    self.available_weights_variations_bycat[cat] += vars

            # By subsample
            if self.has_subsamples:
                for subsample in self.subsamples:
                    weights_by_subsample = self.variations_config["by_subsample"][f"{sample}__{subsample}"]["weights"]
                    for cat, weights in weights_by_subsample.items():
                        for weight in weights:
                            # Ask the WeightsManager the available variations
                            vars = self.weights_manager.get_available_modifiers_byweight(weight)
                            self.available_weights_variations_bysubsample[subsample] += vars
                            self.available_weights_variations_bysubsample_bycat[subsample][cat] += vars
                    
            # Shape variations
            for cat, vars in self.variations_config["shape"].items():
                # Ask the calibrators manager for available variations. 
                # Each calibrator handles the available variations
                for var in vars:
                    variations = self.calibrators_manager.get_available_variations(var)
                    self.available_shape_variations += variations
                    self.available_shape_variations_bycat[cat] += variations

            # shape variations by subsamples
            if self.has_subsamples:
                for subsample in self.subsamples:
                    for cat, vars in self.variations_config["by_subsample"][f"{sample}__{subsample}"]["shape"].items():
                        for var in vars:
                            variations = self.calibrators_manager.get_available_variations(var)
                            self.available_shape_variations_bysubsample[subsample] += variations
                            self.available_shape_variations_bysubsample_bycat[subsample][cat] += variations

        else:  # DATA
            # Add a "weight_variation" nominal for data in each category
            for cat in self.categories_config.keys():
                self.available_weights_variations += ["nominal"]
                self.available_weights_variations_bycat[cat].append("nominal")
                
        # Reduce to set over all the categories
        self.available_weights_variations = set(self.available_weights_variations)
        self.available_shape_variations = set(self.available_shape_variations)
        if self.has_subsamples:
            self.available_weights_variations_bysubsample = {
                sub: set(vars) for sub, vars in self.available_weights_variations_bysubsample.items()
            }
            self.available_shape_variations_bysubsample = {
                sub: set(vars) for sub, vars in self.available_shape_variations_bysubsample.items()
            }
        # Prepare the variations Axes summing all the required variations
        # The variation config is organized as the weights one, by sample and by category, and by subsample
        
        for name, hcfg in deepcopy(hist_config).items():
            # Check if the histogram is active for the current sample
            # We only check for the parent sample, not for subsamples
            if hcfg.only_samples != None:
                if sample not in hcfg.only_samples:
                    continue
            elif hcfg.exclude_samples != None:
                if sample in hcfg.exclude_samples:
                    continue
            # Now we handle the selection of the categories
            cats = []
            for c in self.available_categories:
                if hcfg.only_categories != None:
                    if c in hcfg.only_categories:
                        cats.append(c)
                elif hcfg.exclude_categories != None:
                    if c not in hcfg.exclude_categories:
                        cats.append(c)
                else:
                    cats.append(c)
            # Update the histConf to save the only category
            hcfg.only_categories = list(sorted(cats))
            # Create categories axis
            cat_ax = hist.axis.StrCategory(
                hcfg.only_categories, name="cat", label="Category", growth=False
            )

            # Look over subsamples as we have different variataions for each subsample
            # IF there are no subsamples the subsample == sample
            for subsample in self.subsamples:
                hcfg_sub = deepcopy(hcfg)
                # Variation axes
                if hcfg_sub.variations:
                    # Get all the variation
                    if self.has_subsamples:
                        allvariat = set.union(self.available_weights_variations, self.available_shape_variations,
                                        self.available_weights_variations_bysubsample[subsample],
                                        self.available_shape_variations_bysubsample[subsample])
                    else:
                        allvariat = set.union(self.available_weights_variations, self.available_shape_variations)

                    if hcfg_sub.only_variations != None:
                        # expand wild card and Up/Down
                        only_variations = []
                        for var in hcfg_sub.only_variations:
                            # Check if it is a calibrator name wildcard
                            # an empty string is returned if the calibrator is not found
                            only_variations_calib = self.calibrators_manager.get_available_variations(var)
                            if len(only_variations_calib)>0:
                                only_variations += only_variations_calib
                            else:
                                # Just use the one explicitely asked
                                only_variations.append(var)

                        # filtering the variation list with the available ones
                        allvariat = set(
                            filter(lambda v: v in only_variations or v == "nominal", allvariat)
                        )
                    # sorted is needed to assure to have always the same order for all chunks
                    hcfg_sub.only_variations = list(sorted(set(allvariat)))
                else:
                    hcfg_sub.only_variations = ["nominal"]
                # Defining the variation axis
                var_ax = hist.axis.StrCategory(
                    hcfg_sub.only_variations, name="variation", label="Variation", growth=False
                )

                # Axis in the configuration + custom axes
                if self.isMC:
                    all_axes = [cat_ax, var_ax]
                else:
                    # no variation axis for data
                    all_axes = [cat_ax]
                # the custom axis get included in the hcfg for future use
                hcfg_sub.axes = custom_axes + hcfg_sub.axes
                # Then we add those axes to the full list
                for ax in hcfg_sub.axes:
                    all_axes.append(get_hist_axis_from_config(ax))
                # Creating an histogram object for each subsample
                # Build the histogram object with the additional axes
                hcfg_sub.hist_obj = hist.Hist(
                    *all_axes, storage=hcfg_sub.storage, name="Counts"
                )
                # Save the hist in the configuration and store the full config object
                self.histograms[subsample][name] = hcfg_sub

    def get_histograms(self, subsample):
        # Exclude by default metadata histo
        return {
            key: h.hist_obj
            for key, h in self.histograms[subsample].items()
            if not h.metadata_hist
        }

    def get_metadata_histograms(self, subsample):
        return {
            key: h.hist_obj
            for key, h in self.histograms[subsample].items()
            if h.metadata_hist
        }

    def get_histogram(self, subsample, name):
        return self.histograms[subsample].get(name, None)

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
        import time
        _t_func = time.time()

        # Preload full-sample weights for all categories (MC and data)
        _t0 = time.time()
        weights = {}
        for category in self.available_categories:
            weights[category] = get_weights_by_cat_var(
                self.available_weights_variations_bycat[category],
                self.weights_manager, category, shape_variation,
            )

        # Preload subsample-specific weights (MC only — data has no subsample SFs).
        # weights_sub[subsample][category][variation] holds the subsample weight for
        # the variations explicitly defined for that subsample; all other variations
        # fall back to the nominal subsample weight via dict.get() at fill time.
        _t0 = time.time()
        weights_sub = {}
        if self.has_subsamples and self.isMC:
            for subsample in self.subsamples:
                weights_sub[subsample] = {}
                for category in self.available_categories:
                    avail = set(self.available_weights_variations_bysubsample_bycat[subsample][category]) | {"nominal"}
                    weights_sub[subsample][category] = get_weights_by_cat_var_subsample(
                        avail, self.weights_manager,
                        self.sample + "__" + subsample, category, shape_variation,
                    )
        
        # Cleaning the weights cache decorator between calls.
        self._weights_cache.clear()
        # Looping on the histograms to read the values only once
        # Then categories, subsamples and weights are applied and masked correctly
        # ASSUNTION, the histograms are the same for each subsample
        # we can take the configuration of the first subsample
        _t_axes_total = 0.0
        _t_mask_total = 0.0
        _t_weight_total = 0.0
        _t_fill_total = 0.0
        _n_hists = 0
        _n_cats_sub_combos = 0
        # Per-histogram time tracking: name -> {axes, mask, weight, fill, total, n_fills}
        _per_hist = {}
        # Per-variation time tracking: variation -> total_fill_time
        _per_variation = {}
        # Cache for ak.pad_none results, keyed on (coll, field, pos).
        # Many histograms share the same (coll, field, pos) -- e.g. jet_hists(pos=0)
        # produces JetGood_pt_1, JetGood_eta_1, ... all with pos=0 on JetGood.
        # Caching the pad result avoids re-running ak.pad_none + index 100+ times
        # per chunk for the same collection/field/pos.
        _pad_cache = {}
        for name, histo in self.histograms[self.subsamples[0]].items():
            # logging.info(f"\thisto: {name}")
            if not histo.autofill:
                continue
            if histo.metadata_hist:
                continue  # TODO dedicated function for metadata histograms

            # Check if a shape variation is under processing and
            # if the current histogram does not require that variation for any subsample
            if (
                shape_variation != "nominal"
                and not any(
                    shape_variation in self.histograms[sub][name].hist_obj.axes["variation"]
                    for sub in self.subsamples
                )
            ):
                continue

            _n_hists += 1
            _t_step = time.time()
            _t_hist_total = time.time()
            # Get the filling axes --> without any masking.
            # The flattening has to be applied as the last step since the categories and subsamples
            # work at event level

            fill_categorical = {}
            fill_numeric = {}
            data_ndim = None
            # Per-histogram accumulators (local, summed into _per_hist at the end)
            _ph = {"axes": 0.0, "mask": 0.0, "weight": 0.0, "fill": 0.0, "n_fills": 0}

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

                        if ax.field not in events[ax.coll].fields:
                            ## ToDo. We need to enable skipping some hists, which may not be avialable.
                            ## It could be that some versions of NanoAOD do not contain certain variables, for example, the various Jet Tagger scores
                            ## At the moment, simply `continue` is not enough - it crashes elsewhere
                            ## continue
                            
                            raise ValueError( f"Varible {ax.field} not found in {ax.coll} Collection!")
                        
                        # General collections
                        if ax.pos == None:
                            data = events[ax.coll][ax.field]
                        elif ax.pos >= 0:
                            # Use the per-chunk pad cache to avoid re-running
                            # ak.pad_none + index for the same (coll, field, pos)
                            # across histograms that share the axis.
                            _pad_key = (ax.coll, ax.field, ax.pos)
                            _cached = _pad_cache.get(_pad_key)
                            if _cached is None:
                                _cached = ak.pad_none(
                                    events[ax.coll][ax.field], ax.pos + 1, axis=1
                                )[:, ax.pos]
                                _pad_cache[_pad_key] = _cached
                            data = _cached
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

            _t_axes_total += time.time() - _t_step
            _ph["axes"] = time.time() - _t_hist_total
            _t_step = time.time()
            # Now the variables have been read for all the events
            # We need now to iterate on categories and subsamples
            # Mask the events, the weights and then flatten and remove the None correctly
            for category, cat_mask in categories.get_masks():
                # loop directly on subsamples
                for subsample, subs_mask in subsamples.get_masks():
                    # logging.info(f"\t\tcategory {category}, subsample {subsample}")
                    _n_cats_sub_combos += 1
                    _t_iter = time.time()
                    _t_combo = time.time()
                    mask = cat_mask & subs_mask
                    # Skip empty categories and subsamples
                    if ak.sum(mask) == 0:
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

                    _t_mask_step = time.time()
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
                    # Pre-compute the numpy boolean mask once per (cat, subsample, hist).
                    # The variation loop uses it to select the valid entries from
                    # the per-variation weight arrays. Doing the ak.to_numpy conversion
                    # here (once) instead of inside the inner variation loop (per variation)
                    # removes a recurring small cost; the indexing itself is then plain
                    # numpy fancy-indexing.
                    _isnotnone_np = ak.to_numpy(all_axes_isnotnone)
                    _t_mask_total += time.time() - _t_mask_step
                    _ph["mask"] += time.time() - _t_mask_step

                    # Ok, now we have all the numerical axes with
                    # data that has been masked, flattened
                    # removed the none value --> now we need weights for each variation
                    _t_w_step = time.time()
                    if not histo.no_weights and self.isMC:
                        if shape_variation == "nominal":
                            # ==== HOIST START ====
                            # Precompute the structural broadcast factor ONCE per
                            # (category, subsample, histogram), then reuse it for every
                            # weight variation. This avoids re-running the expensive
                            # ak.ones_like(mask) * weight [mask] flatten chain for
                            # every variation (the original hotspot).
                            #
                            # For mask.ndim==2 (collection-level cut):
                            #   - the final weight = np.repeat(weight[event], counts[event])
                            #     where counts = number-of-True-per-event in mask.
                            #   - We precompute counts and a 0/1 factor that has the
                            #     correct final length, then per variation do a fast
                            #     numpy repeat + multiply.
                            #
                            # For mask.ndim==1 and data_structure.ndim==2:
                            #   The original code computes
                            #     ak.flatten(data_structure * (weight[mask]))
                            #   where weight[mask] is a 1D array of length n_surviving
                            #   and data_structure is 2D jagged of shape (n_surviving, var).
                            #   The result is data_structure.values * weight[event].
                            #
                            #   We tried to hoist this with
                            #     data_structure_flat * np.repeat(weight[mask], counts)
                            #   but hit a shape mismatch in some real-data cases where
                            #   ak.flatten(data_structure) returned fewer elements than
                            #   sum(counts). This appears to happen when data_structure
                            #   contains None / optional elements or when a downstream
                            #   operation (e.g. pad_none) changed its structure. To be
                            #   safe we only enable the 1D+2D hoist when we can verify
                            #   shapes match, and otherwise fall back to the original
                            #   mask_and_broadcast_weight which is correct in all cases.
                            #
                            # For mask.ndim==1 and data is per-event (no data_structure
                            # or ndim==1):
                            #   - the final weight is just weight[mask]. No hoist
                            #     needed; the existing @weights_cache handles it.
                            _bcast_factor = None
                            _bcast_per_event = None  # for mask.ndim==1 + data_structure.ndim==2
                            # Pre-allocated buffer + index for the np.repeat result
                            # (Step 3 of the perf plan). The total output length is
                            # sum(_bcast_per_event); this is the same for every weight
                            # variation, so we allocate it once and refill via
                            # np.take(arr, _repeat_idxs, out=_buf) inside the loop.
                            _repeat_idxs = None
                            _w_out_buf = None
                            _hoist_2d = (mask.ndim == 2)
                            _hoist_1d_2d = False
                            if _hoist_2d:
                                # Per-event count of True values in the mask
                                _bcast_per_event = ak.to_numpy(ak.sum(mask, axis=1))
                                # Pre-allocate the result of np.repeat(weight, counts).
                                # The total length is the sum of per-event object counts.
                                _total_objs = int(_bcast_per_event.sum())
                                # Indices equivalent to np.repeat(np.arange(n_events), counts)
                                _repeat_idxs = np.repeat(
                                    np.arange(len(_bcast_per_event)), _bcast_per_event
                                )
                                _w_out_buf = np.empty(_total_objs, dtype=np.float64)
                            elif mask.ndim == 1 and data_structure is not None and data_structure.ndim == 2:
                                # 1D mask + 2D data_structure hoist (currently disabled
                                # to keep behavior identical to mask_and_broadcast_weight;
                                # the 2D mask hoist above already provides the main win
                                # for the common jet/lepton collection case).
                                # To re-enable, see the long comment above and ensure
                                # shape verification: len(ak.flatten(data_structure))
                                # == int(sum(ak.num(data_structure, axis=1))).
                                _hoist_1d_2d = False
                            # ==== HOIST END ====

                            # if we are working on nominal we fill all the weights variations
                            for variation in self.histograms[subsample][name].hist_obj.axes["variation"]:
                                if variation in self.available_shape_variations or (
                                    self.has_subsamples and
                                    variation in self.available_shape_variations_bysubsample[subsample]
                                ):
                                    # Skip shape variations (full-sample or subsample-specific)
                                    # when processing the nominal shape pass.
                                    continue
                                # Only weights variations, since we are working on nominal sample
                                # Check if this variation exists for this category
                                if variation not in weights[category]:
                                    # it means that the variation is in the axes only
                                    # because it is requested for another category or because the
                                    # variation is by subsample. 
                                    # In this case we fill with the nominal variation
                                    # We get the weights for the current category
                                    weight_varied = weights[category]["nominal"]
                                else:
                                    # We get the weights for the current category
                                    weight_varied = weights[category][variation]

                                # Get subsample-specific weight: use preloaded varied value if
                                # this variation is defined for the subsample, else fall back
                                # to the preloaded nominal subsample weight.
                                weight_sub = (
                                    weights_sub[subsample][category].get(
                                        variation, weights_sub[subsample][category]["nominal"]
                                    )
                                    if self.has_subsamples else 1.
                                )

                                # Broadcast and mask the weight. We use the hoisted
                                # path when the structural factor was precomputed above:
                                #   - _hoist_2d:        output = np.take(weight, idxs) into a
                                #                        pre-allocated buffer (Step 3)
                                #   - _hoist_1d_2d:     output = data_structure_masked_flat * weight_per_event_expanded
                                #                        (currently disabled, see hoist block)
                                #   - else:             fall back to the original mask_and_broadcast_weight
                                #                        (which the @weights_cache may still speed up)
                                _t_w_hoist = time.time()
                                weight_combined = weight_varied * weight_sub
                                # Convert to numpy once per variation
                                if hasattr(weight_combined, "to_numpy"):
                                    _w_np = ak.to_numpy(weight_combined)
                                else:
                                    _w_np = np.asarray(weight_combined)
                                if _hoist_2d:
                                    # Per-event weight expanded to per-object via the
                                    # precomputed index, into the pre-allocated output
                                    # buffer. This avoids allocating a new output array
                                    # for every weight variation.
                                    np.take(_w_np, _repeat_idxs, out=_w_out_buf)
                                    weight_varied = _w_out_buf
                                elif _hoist_1d_2d:
                                    # Per-event weight expanded to per-object using mask counts,
                                    # then elementwise multiplied with the precomputed data_structure
                                    weight_varied = _bcast_factor * np.repeat(_w_np, _bcast_per_event)
                                else:
                                    weight_varied = self.mask_and_broadcast_weight(
                                        category,
                                        subsample,
                                        variation,
                                        weight_combined,
                                        mask,
                                        data_structure,
                                    )
                                _ph["weight_hoist"] = _ph.get("weight_hoist", 0.0) + (time.time() - _t_w_hoist)
                                if custom_weight != None and name in custom_weight:
                                    weight_varied = weight_varied * self.mask_and_broadcast_weight(
                                        category + "customW",
                                        subsample,
                                        variation,
                                        custom_weight[
                                            name
                                        ],  # passing the custom weight to be masked and broadcasted
                                        mask,
                                        data_structure,
                                    )

                                # Then we apply the notnone mask (use the cached numpy version)
                                weight_varied = weight_varied[_isnotnone_np]
                                # Fill the histogram
                                _t_f = time.time()
                                try:
                                    self.histograms[subsample][name].hist_obj.fill(
                                        cat=category,
                                        variation=variation,
                                        weight=weight_varied,
                                        **{**fill_categorical, **fill_numeric_masked},
                                    )
                                except Exception as e:
                                    raise Exception(
                                        f"Cannot fill histogram: {name}, {histo} {e}"
                                    )
                                _t_fill_dt = time.time() - _t_f
                                _t_fill_total += _t_fill_dt
                                _ph["fill"] += _t_fill_dt
                                _ph["n_fills"] += 1
                                _per_variation[variation] = _per_variation.get(variation, 0.0) + _t_fill_dt
                        else:
                            # Check if this shape variation is requested for this category,
                            # either as a full-sample variation or as a subsample-specific one.
                            in_full_sample = shape_variation in self.available_shape_variations_bycat[category]
                            in_subsample   = (self.has_subsamples and
                                              shape_variation in self.available_shape_variations_bysubsample_bycat[subsample][category])
                            if not in_full_sample and not in_subsample:
                                # The variation is in the axis only because it is requested for another
                                # category or another subsample. We cannot fill with nominal here because
                                # the observable will differ under the shape variation.
                                continue
                                
                            # Working on shape variation! only nominal weights
                            # (also using the cache which is cleaned for each shape variation
                            # at the beginning of the function)
                            weight_nom = weights[category]["nominal"]
                            weight_sub = weights_sub[subsample][category]["nominal"] if self.has_subsamples else 1.
                                
                            weight_nom = self.mask_and_broadcast_weight(
                                category,
                                subsample,
                                "nominal",
                                weight_nom * weight_sub,
                                mask,
                                data_structure,
                            )
                            
                            if custom_weight != None and name in custom_weight:
                                weight_nom = weight_nom * self.mask_and_broadcast_weight(
                                    category + "customW",
                                    subsample,
                                    "nominal",
                                    custom_weight[
                                        name
                                    ],  # passing the custom weight to be masked and broadcasted
                                    mask,
                                    data_structure,
                                )
                            # Then we apply the notnone mask (use the cached numpy version)
                            weight_nom = weight_nom[_isnotnone_np]
                            # Fill the histogram
                            _t_f = time.time()
                            try:
                                self.histograms[subsample][name].hist_obj.fill(
                                    cat=category,
                                    variation=shape_variation,
                                    weight=weight_nom,
                                    **{**fill_categorical, **fill_numeric_masked},
                                )
                            except Exception as e:
                                raise Exception(
                                    f"Cannot fill histogram: {name}, {histo} {e}"
                                )
                            _t_fill_dt = time.time() - _t_f
                            _t_fill_total += _t_fill_dt
                            _ph["fill"] += _t_fill_dt
                            _ph["n_fills"] += 1
                            _per_variation[shape_variation] = _per_variation.get(shape_variation, 0.0) + _t_fill_dt
                    ##################################################################################
                    elif not histo.no_weights and not self.isMC:   #DATA
                        # Broadcast and mask the weight (using the cached value if possible)
                        weight_data = weights[category]["nominal"]
                        weight_data = self.mask_and_broadcast_weight(
                            category,
                            subsample,
                            "nominal",
                            weight_data,
                            mask,
                            data_structure,
                        )
                        if custom_weight != None and name in custom_weight:
                            weight_data = weight_data * self.mask_and_broadcast_weight(
                                category + "customW",
                                subsample,
                                "nominal",
                                custom_weight[
                                    name
                                ],  # passing the custom weight to be masked and broadcasted
                                mask,
                                data_structure,
                            )

                        # Then we apply the notnone mask (use the cached numpy version)
                        weight_data = weight_data[_isnotnone_np]
                        # Fill the histogram
                        _t_f = time.time()
                        try:
                            # Data histograms don't have variations but now can be weighted
                            self.histograms[subsample][name].hist_obj.fill(
                                cat=category,
                                weight=weight_data,
                                **{**fill_categorical, **fill_numeric_masked},
                            )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram for Data: {name}, {histo} {e}"
                            )
                        _t_fill_dt = time.time() - _t_f
                        _t_fill_total += _t_fill_dt
                        _ph["fill"] += _t_fill_dt
                        _ph["n_fills"] += 1
                        _per_variation["nominal"] = _per_variation.get("nominal", 0.0) + _t_fill_dt

                    ######################################################
                    elif (
                        histo.no_weights and self.isMC
                    ):  # NO Weights modifier for the histogram
                        _t_f = time.time()
                        try:
                            self.histograms[subsample][name].hist_obj.fill(
                                cat=category,
                                variation="nominal",
                                **{**fill_categorical, **fill_numeric_masked},
                            )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram: {name}, {histo} {e}"
                            )
                        _t_fill_dt = time.time() - _t_f
                        _t_fill_total += _t_fill_dt
                        _ph["fill"] += _t_fill_dt
                        _ph["n_fills"] += 1
                        _per_variation["nominal"] = _per_variation.get("nominal", 0.0) + _t_fill_dt

                    elif histo.no_weights and not self.isMC:
                        # Fill histograms for Data
                        _t_f = time.time()
                        try:
                            self.histograms[subsample][name].hist_obj.fill(
                                cat=category,
                                **{**fill_categorical, **fill_numeric_masked},
                            )
                        except Exception as e:
                            raise Exception(
                                f"Cannot fill histogram: {name}, {histo} {e}"
                            )
                        _t_fill_dt = time.time() - _t_f
                        _t_fill_total += _t_fill_dt
                        _ph["fill"] += _t_fill_dt
                        _ph["n_fills"] += 1
                        _per_variation["nominal"] = _per_variation.get("nominal", 0.0) + _t_fill_dt
                    else:
                        raise Exception(
                            f"Cannot fill histogram: {name}, {histo}, not implemented combination of options"
                        )
                    _t_weight_total += time.time() - _t_w_step
                    _ph["weight"] += time.time() - _t_w_step
            # Commit per-histogram timings
            _ph["total"] = time.time() - _t_hist_total
            _per_hist[name] = _ph

        # Per-histogram fill loop summary
        print(
            f"[TIMING]     [fill_histograms] DONE (shape_variation={shape_variation!r}, "
            f"n_hists={_n_hists}, n_cat_sub_combos={_n_cats_sub_combos}, "
            f"axes_extraction={_t_axes_total:.3f}s, "
            f"mask+flatten={_t_mask_total:.3f}s, "
            f"weight+broadcast={_t_weight_total:.3f}s, "
            f"hist.fill={_t_fill_total:.3f}s, "
            f"total_in_func={time.time()-_t_func:.3f}s, "
            f"pad_cache_size={len(_pad_cache)})"
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



