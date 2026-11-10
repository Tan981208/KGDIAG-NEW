# Dataset access and input preparation

This package contains method code and configuration only. Obtain the datasets from their owners using the links below; do not place downloaded observations, labels, model outputs, or plots in the public code repository. Links were checked on 2026-11-10.

## SWaT-A12

- [iTrust dataset catalogue](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/summary-of-available-datasets/)
- [Official access-request form](https://www.sutd.edu.sg/itrust/request-for-datasets/)
- [Dataset usage terms](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/terms-of-usage/)
- [SWaT release descriptions](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/)

Request **SWaT.A12, March 2026**, including historian observations, signal descriptions, and the accompanying attack-operation documentation. A12 is a release identifier: it does not mean the older A1/A2 December 2015 data. SWaT A4/A5 July 2019 is also a separate release and is not used as a substitute here. The iTrust catalogue lists A12 separately from both older releases.

The A12 historian source used by the project is named `11-Mar-2026_0900_1700.csv`; align retained variable names to the supplied signal descriptions rather than assuming the 51-column schema of older SWaT releases. Preserve invalid-value information and timestamp boundaries during conversion. Supply point labels and documented operation intervals as separate inputs: an attack-run phase annotation and an operation interval are not interchangeable pointwise physical-fault labels.

iTrust requires each recipient to obtain access through its request process and prohibits redistributing its datasets privately or publicly. Consequently, this package supplies access links and code, not a dataset mirror. Acknowledge iTrust as specified in its terms.

## WADI

- [Official WADI descriptions](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/)
- [Release catalogue](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/summary-of-available-datasets/)
- [Access-request form](https://www.sutd.edu.sg/itrust/request-for-datasets/)
- [Usage terms](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/terms-of-usage/)

Record the selected release explicitly. **WADI A1 (October 2017)** and the updated **A2 (November 2019)** have different preparation and annotation files. The A2 description identifies `WADI_14days_new.csv` and `WADI_attackdataLABLE.csv`; its label convention is `-1` for attack and `1` for non-attack. Map labels explicitly to this package's binary convention instead of treating every positive source label as anomalous. Use the matching release's attack table. The December 2023 cross-over run is another separate collection.

Do not concatenate independently recorded periods into one continuous window sequence. Preserve timestamps, normal/attack file membership, recording identifiers, and any invalid-value mask. WADI is subject to the same iTrust access and non-redistribution terms as SWaT.

## SMD (Server Machine Dataset)

- [Original dataset repository](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
- [Authors' dataset description](https://github.com/NetManAIOps/OmniAnomaly#data)
- [Dataset-local license](https://github.com/NetManAIOps/OmniAnomaly/blob/master/ServerMachineDataset/LICENSE)

The original repository provides `train/`, `test/`, `test_label/`, and `interpretation_label/`. Match files by machine identifier, such as `machine-1-1.txt`. The release contains 28 machines with 38 variables each. Treat machines as separate entities; do not create windows across machine boundaries. Training and test files are ordered sequences, while `test_label` contains point labels. Interpretation annotations are a distinct diagnostic input.

The dataset directory carries an MIT license. Retain its notice if redistributing any portion; this package links to the owner instead. Record the repository commit and downloaded file hashes with a run.

## OscGrid

- [Versioned dataset record (v5)](https://doi.org/10.6084/m9.figshare.28465427.v5)
- [Figshare download page (v5)](https://figshare.com/articles/dataset/Open_Dataset_of_Real-World_Oscillograms_from_Electric_Power_Grids/28465427/5)
- [Dataset authors' conversion code](https://github.com/AIRI-Institute/oscgrid)
- [Data descriptor](https://doi.org/10.1038/s41597-026-06587-8)

The authors' code links to version 5; the record also lists later versions. Select a version explicitly and record its DOI and file hashes. The data record uses **CC BY 4.0**, while the authors' code repository uses **MIT**; these are separate licenses.

Follow the authors' conversion instructions for the chosen raw archive. Keep oscillogram identity, within-record sample order, sampling rate, retained channels, missing-channel masks, and the declared mapping from original annotations to binary labels. Treat each oscillogram as an independent recording. Do not construct windows across recording boundaries or insert synthetic normal rows to join records. Continuous channels with different physical units are normalized from the selected normal Fit subset, not from all recordings together.

## Required local preparation record

For each dataset, retain the release identifier, source-file hashes, selected entities/recordings, node order, continuous/categorical schema, explicit normal Fit/Validation/Calibration boundaries, test selection, and annotation meaning. Fit scalers and statistical graph links only on the declared normal data. Validation used to check link stability must also be normal. Test labels are inputs to evaluation, not graph construction, preprocessing selection, score calibration, or threshold selection.

Engineering graph records are a separate input. Each accepted relation must identify its source document, source location, evidence status, endpoints, and type. Dataset access alone does not supply a complete verified process graph. A lag attribute documents a relation; this implementation does not read delayed source features automatically.
