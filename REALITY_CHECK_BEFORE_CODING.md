# BEFORE EVERY CODING SESSION - REALITY CHECK CHECKLIST
## Use this to avoid hallucinations and wrong directions

**Last validated**: April 7, 2026  
**Project**: Interpretable ML System for Indian Hospitals (NOT just mortality prediction)

---

## 🎯 PROJECT SCOPE (MUST MEMORIZE)

**This is NOT**:
- ❌ Just a mortality prediction model
- ❌ A quick ensemble tree model
- ❌ An instantaneous (no temporal data) predictor
- ❌ Building from pre-processed CSV files
- ❌ Aimed for quick hospital deployment

**This IS**:
- ✅ Comprehensive interpretable ML system combining:
  - Real-time ICU data processing (vital signs + lab tests + medications)
  - Multi-organ health tracking (6 organs independently scored)
  - Medicine response tracking (what drugs work, what don't)
  - 24+ hour temporal predictions (patient trajectory modeling)
  - Deep learning backbone (LSTM/GRU + multi-task architecture)
  - Explainability built in (SHAP, feature importance, organ-specific reports)
  - Superior to SOFA/APACHE (clinical validation required)
  - Patient family accessible explanations (not just technical)

---

## 🔍 VALIDATION CHECKPOINT (Answer before coding)

### 1. DATA SOURCE
**Question**: Where are we extracting data FROM?
- [ ] ✅ RAW eICU dataset (`data/raw/eicu/`) ← CORRECT
- [ ] ✅ RAW Challenge2012 dataset (`data/raw/challenge2012/`) ← FOR VALIDATION ONLY
- [ ] ❌ Pre-processed CSVs (`data/processed/`) ← WRONG! (unless Phase 1 is complete)
- [ ] ❌ Numpy arrays (X_24h.npy, etc.) ← WRONG! (outdated intermediate files)

**If you selected ❌**: STOP. Read PROJECT_SCOPE_VALIDATED.md again.

---

### 2. DATA FEATURES
**Question**: What data are we using?
- [ ] Vitals (HR, BP, SpO2, RR, etc.) ← Good start
- [ ] ✅ **Lab tests (147 types)** ← MUST INCLUDE (renal, liver, sepsis markers)
- [ ] ✅ **Medications (75K records with dosage)** ← MUST INCLUDE (treatment tracking)
- [ ] ✅ **SOFA components** ← MUST INCLUDE (organ dysfunction)
- [ ] Nursing notes ← Nice to have
- [ ] ECG data ← Nice to have

**Minimum required**: All ✅ items above

**If missing ANY ✅**: STOP. Add it to feature extraction.

---

### 3. TEMPORAL WINDOW
**Question**: What time window are predictions based on?
- [ ] ❌ Instantaneous (current hour only) ← WRONG - not interpretable
- [ ] ❌ Short-term (6-12 hours) ← WRONG - not enough context
- [ ] ✅ **24+ hours of patient history** ← CORRECT
- [ ] ✅ **Sequential/sliding 24h windows** ← CORRECT approach

**If not 24h windows**: STOP and redesign to use 24h windows.

---

### 4. PREDICTION OUTPUTS
**Question**: What does the model predict?
- [ ] ❌ Just mortality score ← Too narrow
- [ ] ✅ Mortality risk (primary) ← CORRECT
- [ ] ✅ 6 organ health scores (respiratory, CV, renal, hepatic, hematologic, CNS) ← MUST HAVE
- [ ] ✅ Medicine response prediction (will current treatment work?) ← MUST HAVE
- [ ] ✅ Expected recovery trajectory (actual vs predicted 24h ahead) ← MUST HAVE
- [ ] ✅ Feature explanation (SHAP values showing what drives risk) ← MUST HAVE

**Missing ANY ✅**: STOP and add it to model design.

---

### 5. TECHNOLOGY STACK
**Question**: What ML techniques and libraries are we using?

**ML Techniques** (REQUIRED):
- [ ] ✅ Deep Learning backbone (LSTM/GRU or Transformer for temporal) ← REQUIRED
- [ ] ✅ AIML components (multi-task learning, ensemble predictions) ← REQUIRED
- [ ] ✅ Interpretability layer (SOFA rules, SHAP, attention mechanisms) ← REQUIRED
- [ ] Tree ensembles (RF/GB) ← Allowed as secondary component only
- [ ] ❌ Just tree ensembles with no temporal component ← WRONG approach

**Deep Learning Libraries** (REQUIRED):
- [ ] ✅ **PyTorch** (framework) ← PRIMARY
- [ ] ✅ **CUDA 11.8+** (GPU acceleration) ← ESSENTIAL
- [ ] ✅ **cuDNN** (GPU primitives) ← REQUIRED
- [ ] ✅ **Transformers** (Hugging Face - for attention mechanisms) ← FOR MODERN ARCH
- [ ] ✅ **SHAP** (explainability) ← REQUIRED FOR INTERPRETABILITY

**Utility Libraries** (REQUIRED):
- [ ] ✅ **Optuna** or **Ray Tune** (hyperparameter tuning) ← REQUIRED
- [ ] ✅ **Scikit-learn** (preprocessing, baselines) ← REQUIRED
- [ ] ✅ **Weights & Biases** or **TensorBoard** (experiment tracking) ← REQUIRED

**If missing ANY ✅**: STOP and install before proceeding.

---

### 6. PERFORMANCE TARGET
**Question**: What's our performance goal?
- [ ] ❌ "Good enough for quick deployment" ← WRONG mindset
- [ ] ❌ "0.85 AUC is acceptable" ← WRONG threshold
- [ ] ✅ **90+ AUC on mortality prediction** ← NON-NEGOTIABLE
- [ ] ✅ **70%+ recall** at clinical threshold (catch 70% of deaths early) ← REQUIRED
- [ ] ✅ **Surpass SOFA/APACHE performance** ← CLINICAL VALIDATION

**If lower targets**: STOP. We need 90+ AUC or this isn't good enough.

---

### 7. EXPLAINABILITY
**Question**: Can patient families understand the output?
- [ ] ❌ Technical feature importance jargon ← NOT SUFFICIENT
- [ ] ❌ Just AUC score ← NOT USEFUL
- [ ] ✅ **Organ-specific health scores (0-10 per organ)** ← NEEDED
- [ ] ✅ **Top 3-5 factors driving risk** (in plain language) ← NEEDED
- [ ] ✅ **Progress vs expected (text + chart)** ← NEEDED
- [ ] ✅ **Medicine response explanation** ← NEEDED

**If can't explain to family easily**: STOP and redesign output.

---

### 8. EVALUATION DATA
**Question**: Are we using appropriate train/test/validation splits?
- [ ] ✅ eICU primary evaluation (2,520 patients, 5% mortality) ← MAIN
- [ ] ✅ Challenge2012 external validation (12,000 patients) ← CROSS-CHECK
- [ ] ✅ Temporal split (chronological) ← BETTER than random
- [ ] ✅ Stratified split (by mortality)← NEEDED for imbalanced class
- [ ] ❌ No cross-validation ← NOT ACCEPTABLE
- [ ] ❌ Random split only ← RISK OF DATA LEAKAGE

**Missing protection against leakage**: STOP and fix.

---

### 9. DOCUMENTATION
**Question**: Can we trace back decisions?
- [ ] ✅ Feature engineering logic documented ← REQUIRED
- [ ] ✅ Data preprocessing steps recorded ← REQUIRED
- [ ] ✅ Model architecture decisions justified ← REQUIRED
- [ ] ✅ Hyperparameter choices explained ← REQUIRED
- [ ] ✅ Validation methodology clear ← REQUIRED

**Missing documentation**: STOP and document before proceeding.

---

## 🚨 RED FLAGS (STOP IMMEDIATELY IF YOU SEE THESE)

1. **"Let's just use pre-processed CSVs"**
   - ❌ Wrong - we need raw data
   - ✅ Do: Trace back to raw eICU CSVs

2. **"We don't need medication data"**
   - ❌ Wrong - medicine tracking is core requirement
   - ✅ Do: Extract admissiondrug.csv, medication.csv

3. **"Instant prediction is fine"**
   - ❌ Wrong - need 24h context for interpretability
   - ✅ Do: Create 24h sliding windows

4. **"Tree ensemble achieved good AUC, let's deploy"**
   - ❌ Wrong - need AIML + deep learning AND 90+ AUC
   - ✅ Do: Add LSTM/transformer, improve to 90+

5. **"Just predict mortality, not organ dysfunction"**
   - ❌ Wrong - organ tracking is key differentiator
   - ✅ Do: Build 6-organ health scores

6. **"Output is just AUC number"**
   - ❌ Wrong - families won't understand
   - ✅ Do: Build patient-friendly explanations

7. **"We haveold models at 0.88+ AUC, good enough"**
   - ❌ Wrong - target is 90+ AUC
   - ✅ Do: Improve to 90+ before deployment

8. **"Let's skip Challenge2012, just use eICU"**
   - ❌ Wrong - need external validation
   - ✅ Do: Validate on Challenge2012 too

---

## ✅ GO/NO-GO DECISION MATRIX

| **Technology** | Go ✅ | No-Go ❌ |
|---|-------|---------|
| **Data Source** | RAW eICU CSVs | Pre-processed, cached arrays |
| **Features** | 200+ from labs+vitals+meds | <100, vitals only |
| **Temporal Window** | 24+ hours | Instant/hourly |
| **Predictions** | Mortality + 6 organs + response | Only mortality |
| **DL Framework** | PyTorch + LSTM/Transformer | TensorFlow only or no DL |
| **GPU/CUDA** | CUDA 11.8+ verified working | No GPU or CUDA not working |
| **Interpretability** | SHAP + organ rules + family text | Technical only |
| **Target AUC** | 90+ | <90 |
| **Validation** | eICU + Challenge2012 | Single dataset |

**Decision**: If ANY ❌ row is TRUE → DO NOT PROCEED. Fix first.

---

## 🔄 BEFORE EACH CODING SESSION

**Step 1: Read this checklist** (2 mins)

**Step 2: Answer the 9 validation questions above** (5 mins)

**Step 3: Check for red flags** (2 mins)

**Step 4: Verify project scope vs what you're about to code** (3 mins)

**If ANY misalignment detected**:
- STOP
- Read PROJECT_SCOPE_VALIDATED.md
- Align code plan with project scope
- THEN proceed

---

## 📊 QUICK REFERENCE - WHAT WE HAVE

| Item | Details | Location |
|------|---------|----------|
| **eICU Dataset** | 2,520 patients, 1.6M vitals, 434K labs, 75K meds | `data/raw/eicu/` |
| **Challenge2012** | 12,000 patients, sparse data | `data/raw/challenge2012/` |
| **Target Output** | Mortality AUC ≥ 90%, 6-organ health scores | Unknown yet |
| **Temporal Unit** | 24-hour windows | Not yet implemented |
| **Feature Count** | Target 200+ from all sources | Under construction |
| **Model Type** | LSTM + multi-task + interpretability layer | Not yet built |
| **Explainability** | SHAP + organ rules + family-friendly text | Not yet implemented |

---

## 💡 WHEN IN DOUBT, REMEMBER

> "We're not building just another mortality predictor. We're building a comprehensive, explainable system that helps Indian hospitals make better decisions and helps patient families understand what's happening. It must use 24+ hours of data, track all organs, explain medicine responses, and be better than SOFA/APACHE. Nothing less."

---

**Questions?** Re-read PROJECT_SCOPE_VALIDATED.md or PHASE1_START_HERE.md

**Ready to code?** Make sure you answer all 9 validation questions with ✅
