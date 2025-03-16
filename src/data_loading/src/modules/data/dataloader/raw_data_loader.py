from collections import defaultdict
from pylidc.utils import consensus as pylidc_utils_consensus
from torch.utils.data import Dataset
import numpy
import os
import pylidc

from src.modules.data import data_transformations


class LIDCIDRIRawDataLoader(Dataset):
    def __init__(self, raw_data_dir_path, raw_data_transforms=None):
        self._raw_data_dir_path = raw_data_dir_path
        self._raw_data_transforms = raw_data_transforms
        self._subject_ids = sorted([
            subject_id for subject_id in os.listdir(raw_data_dir_path)
            if "LIDC-IDRI" in subject_id
        ])

    def __len__(self):
        return len(self._subject_ids)

    def __getitem__(self, idx):
        pylidc_scan = pylidc.query(pylidc.Scan).filter(
            pylidc.Scan.patient_id == self._subject_ids[idx]
        ).first()
        dicom_img_volume = \
            self.get_dicom_img_volume(pylidc_scan, transform_data=True)
        dicom_img_metadata = \
            self.get_dicom_img_metadata(pylidc_scan, dicom_img_volume)
        return dicom_img_volume, dicom_img_metadata

    def get_dicom_img_metadata(self, pylidc_scan, dicom_img_volume):
        dicom_img_raw_volume = self.get_dicom_img_volume(pylidc_scan)
        dicom_img_metadata = dict(
            id=dict(
                patient_id=pylidc_scan.patient_id,
                series_instance_uid=pylidc_scan.series_instance_uid,
                study_instance_uid=pylidc_scan.study_instance_uid),
            attribute=dict(
                pixel_spacing=dict(
                    raw=pylidc_scan.pixel_spacing,
                    transformed=self._raw_data_transforms.pixel_spacing),
                slice_thickness=dict(
                    raw=pylidc_scan.slice_thickness,
                    transformed=self._raw_data_transforms.slice_thickness),
                pixels_values=dict(
                    raw=tuple([
                        round(dicom_img_raw_volume.min(), 2),
                        round(dicom_img_raw_volume.max(), 2)
                    ]),
                    transformed=tuple([
                        round(dicom_img_volume.min(), 2),
                        round(dicom_img_volume.max(), 2)
                    ])
                ),
                axis_transposition_protocol=tuple(
                    self._raw_data_transforms.axis_transposition_protocol
                ),
                flip_axis=self._raw_data_transforms.flip_axis,
                shape=dict(
                    raw=dicom_img_raw_volume.shape,
                    transformed=dicom_img_volume.shape,
                    rescaling_factor=tuple(
                        numpy.array(dicom_img_volume.shape) / numpy.array(
                            numpy.transpose(
                                dicom_img_raw_volume,
                                axes=self._raw_data_transforms.axis_transposition_protocol
                            ).shape
                        )
                    )
                )
            )
        )
        return dicom_img_metadata

    def get_dicom_img_volume(self, pylidc_scan, transform_data=False):
        dicom_img_volume = pylidc_scan.to_volume(
            raw_data_dir=self._raw_data_dir_path,
            verbose=False
        )
        if self._raw_data_transforms and transform_data:
            dicom_img_volume = numpy.transpose(
                numpy.array(dicom_img_volume),
                axes=self._raw_data_transforms.axis_transposition_protocol
            )
            dicom_img_volume = data_transformations.resample_volume(
                volume=dicom_img_volume,
                volume_slice_thickness=dict(
                    input=pylidc_scan.slice_thickness,
                    output=self._raw_data_transforms.slice_thickness
                ),
                volume_pixel_spacing=dict(
                    input=pylidc_scan.pixel_spacing,
                    output=self._raw_data_transforms.pixel_spacing
                )
            )
            dicom_img_volume = data_transformations.flip_volume(
                volume=dicom_img_volume,
                axis=self._raw_data_transforms.flip_axis
            )
            dicom_img_volume = data_transformations.clip_volume_values(
                volume=dicom_img_volume,
                min_max_clip_values=self._raw_data_transforms.min_max_clip_hu_values
            )
            dicom_img_volume = data_transformations.rescale_volume_values(
                dicom_img_volume,
                output_min_max_value_range=self._raw_data_transforms.min_max_value_range
            )
        return dicom_img_volume

    @staticmethod
    def get_nodule_annotation_consensuses(
            nodule_annotation_clusters,
            raw_dicom_img_shape,
            transformed_dicom_img_shape
    ):
        """Get the nodule annotation consensuses.

        Args:
          pylidc_scan:
            A pylidc.Scan.Scan class that holds some of the DICOM attributes
            associated with a patient's CT scan that was queried from the LIDC
            dataset.

        Variables:
          consensus_mask:
          consensus_bounding_box:
            Tuple of slices corresponding to the nodule bounding box indices.
            This can be used to easily index into the NumPy CT image volume
            (see https://pylidc.github.io/tuts/annotation.html).
          masks:

        Returns:
          The nodule annotation consensuses.
        """

        rescaling_factor = tuple(
            numpy.array(transformed_dicom_img_shape)
            /numpy.array(raw_dicom_img_shape)
        )

        nodule_annotation_consensuses = dict()
        for nodule_index, nodule_annotations in enumerate(
                nodule_annotation_clusters, 1
        ):
            consensus_mask, consensus_bounding_box, masks = \
                pylidc_utils_consensus(nodule_annotations, clevel=0.5)

            # Calculate the mean value of nodule centroid coordinates
            # and median malignancy of nodules
            nodule_centroid_coordinates_annotations = []
            nodule_characteristic_annotations = dict(
                semantic=defaultdict(list),
                geometric=defaultdict(list)
            )
            for nodule_annotation in nodule_annotations:
                nodule_centroid_coordinates = [
                    nodule_annotation.centroid[2],
                    nodule_annotation.centroid[0],
                    nodule_annotation.centroid[1]
                ]
                nodule_centroid_coordinates = \
                    numpy.array(nodule_centroid_coordinates) * rescaling_factor
                nodule_centroid_coordinates[1] = (
                    transformed_dicom_img_shape[1]
                    - nodule_centroid_coordinates[1]
                )
                nodule_centroid_coordinates_annotations.append(
                    nodule_centroid_coordinates
                )

                nodule_characteristic_annotations['semantic'][
                    'subtlety'
                ].append(nodule_annotation.subtlety)
                nodule_characteristic_annotations['semantic'][
                    'internalStructure'
                ].append(nodule_annotation.internalStructure)
                nodule_characteristic_annotations['semantic'][
                    'calcification'
                ].append(nodule_annotation.calcification)
                nodule_characteristic_annotations['semantic'][
                    'sphericity'
                ].append(nodule_annotation.sphericity)
                nodule_characteristic_annotations['semantic'][
                    'margin'
                ].append(nodule_annotation.margin)
                nodule_characteristic_annotations['semantic'][
                    'lobulation'
                ].append(nodule_annotation.lobulation)
                nodule_characteristic_annotations['semantic'][
                    'spiculation'
                ].append(nodule_annotation.spiculation)
                nodule_characteristic_annotations['semantic'][
                    'texture'
                ].append(nodule_annotation.texture)
                nodule_characteristic_annotations['semantic'][
                    'malignancy'
                ].append(nodule_annotation.malignancy)
                nodule_characteristic_annotations['geometric'][
                    'diameter'
                ].append(nodule_annotation.diameter)
                nodule_characteristic_annotations['geometric'][
                    'surface_area'
                ].append(nodule_annotation.surface_area)
                nodule_characteristic_annotations['geometric'][
                    'volume'
                ].append(nodule_annotation.volume)

            mean_value_of_nodule_centroid_coordinates = tuple(
                numpy.mean(nodule_centroid_coordinates_annotations, axis=0)
            )
            median_of_nodule_characteristic_annotations = {
                characteristic_type: {
                    characteristic: numpy.median(annotation_value)
                    for characteristic, annotation_value
                    in nodule_characteristic_annotations[
                        characteristic_type
                    ].items()
                }
                for characteristic_type
                in nodule_characteristic_annotations.keys()
            }

            nodule_annotation_consensuses[f'nodule_{nodule_index}'] = dict(
                bounding_box=consensus_bounding_box,
                mask=numpy.transpose(
                    numpy.array(consensus_mask),
                    axes=(2, 0, 1)
                ),
                centroid_coordinate=mean_value_of_nodule_centroid_coordinates,
                characteristics=median_of_nodule_characteristic_annotations)
        return nodule_annotation_consensuses
