# Benchmark Research for DesignPower Validation

## Methodology for Establishing Authoritative Benchmarks

### Criteria for Authoritative Sources

1. **Primary Sources:** Original methodology papers
2. **Textbook Standards:** Widely-cited statistical textbooks 
3. **Software Documentation:** Official manuals with worked examples
4. **Regulatory Guidance:** FDA, EMA guidelines with examples
5. **Cross-Validation:** Multiple independent sources showing same result

### Research Process

1. **Identify canonical papers** for each design type
2. **Extract worked examples** with complete parameter specifications
3. **Verify calculations** using multiple methods when possible
4. **Document assumptions** and approximations used
5. **Cross-reference** against other authoritative sources

---

## PARALLEL RCT - BINARY OUTCOMES

### Primary Sources to Research

#### 1. Fleiss, Cohen & Everitt (1973) "Statistical Methods for Rates and Proportions"
- **Status:** Need to locate specific examples
- **Target:** Chapter on sample size determination
- **Expected:** Worked examples with p1, p2, alpha, power → n

#### 2. Lachin (1981) "Introduction to sample size determination"
- **Citation:** Controlled Clinical Trials, 2:93-113
- **Status:** Need to access paper
- **Expected:** Binary outcome formulas with examples

#### 3. Machin, Campbell, Tan & Tan (2018) "Sample Sizes for Clinical, Laboratory and Epidemiological Studies"
- **Status:** Need to check 4th edition examples
- **Expected:** Multiple worked examples across chapters

#### 4. R Documentation - `pwr.2p.test()`
- **Status:** Need to document exact examples from help files
- **Advantage:** Widely used, well-tested implementation

### Research Questions to Answer

1. **Which test statistic?** (Chi-square, Fisher's exact, Normal approximation)
2. **Continuity correction?** (When applied, impact on sample size)
3. **Effect size measure?** (Risk difference, odds ratio, Cohen's h)
4. **Allocation ratios?** (Equal vs unequal group sizes)

---

## PARALLEL RCT - CONTINUOUS OUTCOMES

### Primary Sources to Research

#### 1. Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
- **Status:** Need Table 2.3.1 and related examples
- **Target:** Effect sizes d=0.2, 0.5, 0.8 with corresponding sample sizes
- **Standard:** Most widely cited power analysis reference

#### 2. Julious (2004) "Sample sizes for clinical trials with Normal data"
- **Citation:** Statistics in Medicine, 23:1921-1986
- **Status:** Need to access comprehensive review
- **Expected:** Detailed examples for various scenarios

#### 3. Chow, Shao & Wang (2008) "Sample Size Calculations in Clinical Research"
- **Status:** Need specific worked examples
- **Expected:** Chapter on parallel group designs

### Key Parameters to Document

1. **Effect size definitions** (standardized mean difference)
2. **Variance assumptions** (equal vs unequal variances)
3. **Test types** (t-test vs Welch test vs Mann-Whitney)
4. **Sided-ness** (one-sided vs two-sided tests)

---

## PARALLEL RCT - SURVIVAL OUTCOMES

### Primary Sources to Research

#### 1. Schoenfeld (1981) "The asymptotic properties of nonparametric tests for comparing survival distributions"
- **Citation:** Biometrika, 68:316-319
- **Status:** Need original paper for log-rank test formulas
- **Expected:** Theoretical foundation with examples

#### 2. Lachin & Foulkes (1986) "Evaluation of sample size and power for analyses of survival"
- **Citation:** Statistics in Medicine, 5:391-413
- **Status:** Need to access for comprehensive methodology
- **Expected:** Multiple survival scenarios

#### 3. Collett (2015) "Modelling Survival Data in Medical Research"
- **Status:** Need 3rd edition examples
- **Expected:** Worked examples for log-rank test sample sizes

### Specific Scenarios to Document

1. **Proportional hazards** with different hazard ratios
2. **Exponential survival** with different medians
3. **Accrual periods** and follow-up effects
4. **Loss to follow-up** considerations

---

## CLUSTER RANDOMIZED TRIALS

### Primary Sources to Research

#### 1. Donner & Klar (2000) "Design and Analysis of Cluster Randomization Trials in Health Research"
- **Status:** Need worked examples from Chapters 3-5
- **Expected:** Binary and continuous outcomes with ICC effects
- **Authority:** Definitive textbook on cluster trials

#### 2. Campbell, Piaggio, Elbourne & Altman (2012) "CONSORT 2010 statement: extension to cluster randomised trials"
- **Citation:** BMJ, 345:e5661
- **Status:** Need to check for sample size examples
- **Expected:** Methodological guidance with examples

#### 3. Hemming, Girling, Sitch et al. (2011) "Sample size calculations for cluster randomised controlled trials"
- **Citation:** BMJ, 343:d5156
- **Status:** Need comprehensive examples
- **Expected:** Multiple design scenarios

### Critical Parameters to Document

1. **ICC values** for different outcomes and settings
2. **Cluster size variations** (fixed vs variable)
3. **Design effects** calculations and interpretations
4. **Matching strategies** and their sample size implications

---

## SINGLE ARM TRIALS

### Primary Sources to Research

#### 1. A'Hern (2001) "Sample size tables for exact single-stage phase II designs"
- **Citation:** Statistics in Medicine, 20:859-866
- **Status:** Need complete Table 1 documentation
- **Expected:** Exact binomial calculations for various p0, p1 combinations

#### 2. Simon (1989) "Optimal two-stage designs for phase II clinical trials"
- **Citation:** Controlled Clinical Trials, 10:1-10
- **Status:** Need Tables 1-3 with complete specifications
- **Expected:** Minimax and optimal designs

#### 3. Fleming (1982) "One-sample multiple testing procedure for phase II clinical trials"
- **Citation:** Biometrics, 38:143-151
- **Status:** Need original methodology and examples
- **Expected:** Single-stage exact tests

### Design Variants to Document

1. **Single-stage designs** (A'Hern, exact binomial)
2. **Two-stage designs** (Simon's minimax and optimal)
3. **Continuous outcomes** (one-sample t-tests)
4. **Survival outcomes** (exponential assumptions)

---

## RESEARCH ACTION PLAN

### Phase 1: Literature Collection (Priority)
- [ ] Access primary papers through institutional library
- [ ] Document complete bibliographic information
- [ ] Scan for worked examples and tables
- [ ] Create digital repository of key pages/tables

### Phase 2: Example Extraction
- [ ] Create standardized format for documenting examples
- [ ] Extract all parameter values and assumptions
- [ ] Record exact calculation methods used
- [ ] Note any approximations or simplifications

### Phase 3: Cross-Validation
- [ ] Compare examples across multiple sources
- [ ] Identify discrepancies and investigate causes
- [ ] Select most authoritative version when conflicts exist
- [ ] Document rationale for benchmark selection

### Phase 4: Software Cross-Check
- [ ] Verify examples against R, SAS, Stata
- [ ] Document any software-specific variations
- [ ] Identify reference implementations to emulate
- [ ] Record version information for reproducibility

---

## DOCUMENTATION TEMPLATE

For each benchmark, document:

```markdown
### Benchmark: [Descriptive Name]

**Source:** Author (Year). Title. Journal, Volume:Pages.
**Page/Table:** Specific location in source
**Example Number:** If multiple examples in source

**Parameters:**
- Parameter 1: value (units, if applicable)
- Parameter 2: value
- [etc.]

**Assumptions:**
- Statistical test used
- Approximations made
- Continuity corrections
- Sided-ness of test

**Expected Result:**
- Primary outcome: value ± precision
- Secondary outcomes: values

**Cross-References:**
- Source 2: [citation] - confirms/differs
- Software X: [version] - produces [result]

**Notes:**
- Any special considerations
- Known variations in methodology
- Historical context if relevant
```

---

## SUCCESS METRICS

### Completeness Targets
- **Parallel RCT Binary:** ≥5 authoritative benchmarks
- **Parallel RCT Continuous:** ≥5 authoritative benchmarks  
- **Parallel RCT Survival:** ≥3 authoritative benchmarks
- **Cluster RCT:** ≥4 authoritative benchmarks (binary + continuous)
- **Single Arm:** ≥4 authoritative benchmarks

### Quality Standards
- Each benchmark verified against ≥2 independent sources
- Complete parameter specifications documented
- Calculation methodology clearly described
- Software cross-validation performed
- Tolerance levels justified based on source precision

This research foundation will ensure DesignPower validation is based on the most authoritative and widely-accepted standards in the field.