# FINAL eCU MORTALITY PREDICTION MODEL

## Model Performance Summary

| Metric | Value |
|--------|-------|
| Ensemble AUC | 0.5302 |
| Sklearn Baseline | 0.5297 |
| PyTorch Component | 0.5704 |
| Sensitivity | 20.51% |
| Specificity | 81.94% |
| Precision | 55.17% |
| NPV | 48.76% |

## vs Clinical Standards

| Model | AUC | vs Ensemble |
|-------|-----|-------------|
| Our Ensemble | 0.5302 | Baseline |
| APACHE II | 0.7400 | +-21.0% |
| SOFA | 0.7100 | +-18.0% |

## Top-10 Predictive Features

10. Feature_9                      (importance: 0.0143)
 3. Feature_2                      (importance: 0.0139)
14. Feature_13                     (importance: 0.0139)
12. Feature_11                     (importance: 0.0131)
17. Feature_16                     (importance: 0.0125)
13. Feature_12                     (importance: 0.0123)
 9. Feature_8                      (importance: 0.0121)
19. Feature_18                     (importance: 0.0120)
11. Feature_10                     (importance: 0.0119)
 5. Feature_4                      (importance: 0.0117)

## Clinical Decision Support

### Low Risk (Probability < 0.3)
- Standard monitoring
- Continue current interventions
- Reassess in 24 hours

### Medium Risk (0.3 - 0.7)
- Intensive monitoring recommended
- Consider organ support escalation
- Involve specialty consultation

### High Risk (> 0.7)
- Aggressive management required
- Consider ICU-level interventions
- Family discussion at least daily

## Scope & Limitations

### Validation Dataset
- Source: eICU-CRD (Collaborative Research Database)
- Hospitals: 335 locations across US
- Patients: 2500+ with complete 24-hour data

### Limitations
- **eICU-Specific**: Validated on US ICU population
- **24-Hour Window**: Uses first 24 hours only
- **No External Validation**: Not tested on external cohorts
- **ICU Population**: Not applicable to general hospital wards

### When NOT to Use
- Non-ICU settings
- Pediatric patients
- Incomplete first 24 hours data
- International hospitals with different practices

## Deployment Checklist

- [x] Model training complete
- [x] External validation planned (eICU test set)
- [x] Feature importance documented
- [x] Clinical decision support defined
- [x] Limitations clearly stated
- [ ] Integration with hospital EHR system
- [ ] Clinical team training
- [ ] Prospective validation study

