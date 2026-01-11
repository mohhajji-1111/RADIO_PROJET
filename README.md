# NSCLC-Radiomics Dataset Analyzer - Quick Start Guide

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pydicom pandas numpy SimpleITK
```

### Step 2: Configure Paths

Open `nsclc_dataset_analyzer.py` and verify these paths:

```python
DATASET_ROOT = r"C:\Users\HP\Desktop\RADIO_PROJET\DATA\NSCLC-Radiomics"
OUTPUT_DIR = r"C:\Users\HP\Desktop\RADIO_PROJET"
```

### Step 3: Run the Analyzer

```bash
python nsclc_dataset_analyzer.py
```

### Step 4: View Results

The script generates two CSV files:
- `nsclc_metadata_extracted.csv` - Complete metadata for all series
- `nsclc_quality_checks.csv` - Quality validation results

---

## 📊 What Does This Script Do?

1. ✅ Scans all patient folders in the dataset
2. ✅ Identifies CT series and RTSTRUCT series
3. ✅ Extracts comprehensive DICOM metadata
4. ✅ Performs quality checks (missing data detection)
5. ✅ Generates summary statistics
6. ✅ Exports results to CSV files

---

## 🎯 Key Features

### CT Metadata Extracted:
- Slice thickness, pixel spacing
- Number of slices, image dimensions
- Manufacturer, scanner model
- Study dates, series UIDs
- KVP, convolution kernel

### RTSTRUCT Metadata Extracted:
- Structure set information
- ROI names and count
- Referenced frame of reference
- Manufacturer information

### Quality Checks:
- ⚠️ Missing CT series
- ⚠️ Missing RTSTRUCT series
- ⚠️ Multiple series (anomalies)

---

## 💡 Customization

### Analyze Specific Patients Only

Edit the `main()` function:

```python
# Change this line:
analyzer.analyze_all_patients(all_patients[:5])  # First 5 patients

# To analyze all patients:
analyzer.analyze_all_patients()  # All patients

# To analyze specific patients:
specific_patients = ['LUNG1-001', 'LUNG1-005', 'LUNG1-010']
analyzer.analyze_all_patients(specific_patients)
```

### Add Custom Metadata Fields

Extend the `analyze_ct_series()` method:

```python
metadata = {
    # ... existing fields ...
    'YourCustomField': ds.get('YourDICOMTag', 'N/A'),
}
```

---

## 📖 For Full Documentation

See `PHASE1_DOCUMENTATION.md` for:
- Complete dataset structure explanation
- DICOM hierarchy details
- Detailed method descriptions
- Troubleshooting guide

---

## 🎯 Testing (First Run)

For initial testing, the script is configured to analyze **first 5 patients only**.

To analyze the complete dataset:
1. Open `nsclc_dataset_analyzer.py`
2. Find line: `analyzer.analyze_all_patients(all_patients[:5])`
3. Change to: `analyzer.analyze_all_patients()`
4. Run again

**Note**: Full dataset analysis may take 10-30 minutes depending on system performance.

---

## ✅ Expected Output Structure

```
RADIO_PROJET/
├── nsclc_dataset_analyzer.py          # Main script
├── requirements.txt                   # Dependencies
├── PHASE1_DOCUMENTATION.md           # Full documentation
├── README.md                         # This file
├── nsclc_metadata_extracted.csv      # Generated: All metadata
└── nsclc_quality_checks.csv          # Generated: Quality checks
```

---

## 🐛 Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pydicom'`  
**Fix**: Run `pip install pydicom pandas numpy`

**Issue**: CSV file is empty  
**Fix**: Check that DATASET_ROOT path is correct

**Issue**: Script is slow  
**Fix**: Normal for large datasets. Start with subset (first 5-10 patients)

---

## 📞 Next Steps

After Phase 1 completion, you can proceed to:
- **Phase 2**: Load 3D CT volumes and RTSTRUCT masks
- **Phase 3**: Extract radiomics features
- **Phase 4**: Machine learning model training

---

**Happy Analyzing! 🏥📊**
