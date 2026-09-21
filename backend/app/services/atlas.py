"""Atlas mapping (FR-6).

[REPORT] atlas registration converts voxel-level lesion coordinates into
anatomical labels. [CHOICE] Harvard-Oxford atlas + SimpleITK rigid/affine.
Harvard-Oxford is cortical; the 'MCA territory' dashboard label comes from a
vascular-territory lookup over atlas regions [CHOICE].
"""
import numpy as np

# [CHOICE] vascular territory lookup: cortical HO regions -> likely territory
MCA_REGIONS = {
    "Insular Cortex",
    "Frontal Operculum Cortex",
    "Central Opercular Cortex",
    "Parietal Operculum Cortex",
    "Planum Polare",
    "Heschl's Gyrus",
    "Planum Temporale",
    "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis",
    "Precentral Gyrus",
    "Postcentral Gyrus",
    "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division",
    "Angular Gyrus",
}


def territory_for(region: str) -> str:
    return "MCA" if region in MCA_REGIONS else "Other/vertebrobasilar"


class Atlas:
    def __init__(self, load=True):
        self.img = None
        self.labels = ["Background"]
        if load:
            try:
                import nibabel as nib
                from nilearn import datasets

                ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
                self.img = ho.maps if hasattr(ho.maps, "get_fdata") else nib.load(ho.maps)
                self.labels = list(ho.labels)
            except Exception:
                self.img = None  # offline/tests: mapping falls back gracefully

    def get_region_info(self, region_id):
        if 0 <= region_id < len(self.labels):
            return self.labels[region_id]
        return f"Region {region_id}"


def register_to_mni(moving_path, mni_template):
    import SimpleITK as sitk

    fixed = sitk.ReadImage(mni_template, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(50)  # multimodal-safe
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetInitialTransform(
        sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        ),
        inPlace=False,
    )
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    return r.Execute(fixed, moving)


def map_to_atlas(mask_mni: np.ndarray, atlas: Atlas):
    if atlas.img is None:
        xs = np.where(mask_mni > 0)[0]
        hemi = "Left" if (len(xs) and xs.mean() < mask_mni.shape[0] / 2) else "Right"
        return {"hemisphere": hemi, "regions": [], "territory": "Unknown"}
    a = np.asarray(atlas.img.dataobj).astype(int)
    ids, counts = np.unique(a[mask_mni > 0], return_counts=True)
    total = counts.sum() + 1e-8
    regions = [
        {"region": atlas.labels[i], "overlap_pct": round(100 * c / total, 1)}
        for i, c in sorted(zip(ids, counts), key=lambda t: -t[1])
        if i > 0
    ]
    # hemisphere from x-coordinate relative to midline in MNI space
    xs = np.where(mask_mni > 0)[0]
    hemi = "Left" if xs.mean() < mask_mni.shape[0] / 2 else "Right"  # verify affine sign
    top = regions[0]["region"] if regions else ""
    return {"hemisphere": hemi, "regions": regions, "territory": territory_for(top)}
