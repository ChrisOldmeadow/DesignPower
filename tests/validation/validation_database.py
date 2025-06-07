"""
Validation database and reporting system for DesignPower.

This module provides comprehensive validation tracking, documentation,
and reporting against established gold standards.
"""

import json
import sqlite3
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib


@dataclass
class ValidationSource:
    """Documentation for a validation source."""
    source_id: str
    title: str
    authors: str
    year: int
    publisher: str
    isbn: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    page_reference: str = ""
    table_figure: str = ""
    notes: str = ""
    authority_level: str = "high"  # high, medium, low
    source_type: str = "textbook"  # textbook, paper, software, regulatory


@dataclass
class ValidationBenchmark:
    """A specific validation benchmark from a source."""
    benchmark_id: str
    source_id: str
    example_name: str
    design_type: str
    outcome_type: str
    calculation_type: str
    test_method: str
    parameters: Dict[str, Any]
    expected_results: Dict[str, Any]
    tolerance: float
    assumptions: List[str]
    notes: str = ""
    verified_by: List[str] = None  # Cross-validation sources


@dataclass
class ValidationResult:
    """Result of running a validation benchmark."""
    result_id: str
    benchmark_id: str
    timestamp: str
    designpower_version: str
    passed: bool
    actual_results: Dict[str, Any]
    comparisons: Dict[str, Dict[str, Any]]
    execution_time: float
    error_message: str = ""
    warnings: List[str] = None


class ValidationDatabase:
    """Database for tracking validation benchmarks and results."""
    
    def __init__(self, db_path: str = "tests/validation/validation.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the validation database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    publisher TEXT,
                    isbn TEXT,
                    doi TEXT,
                    url TEXT,
                    page_reference TEXT,
                    table_figure TEXT,
                    notes TEXT,
                    authority_level TEXT,
                    source_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    benchmark_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    example_name TEXT NOT NULL,
                    design_type TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    calculation_type TEXT NOT NULL,
                    test_method TEXT,
                    parameters TEXT NOT NULL,  -- JSON
                    expected_results TEXT NOT NULL,  -- JSON
                    tolerance REAL NOT NULL,
                    assumptions TEXT,  -- JSON array
                    notes TEXT,
                    verified_by TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources (source_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    benchmark_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    designpower_version TEXT,
                    passed BOOLEAN NOT NULL,
                    actual_results TEXT NOT NULL,  -- JSON
                    comparisons TEXT NOT NULL,  -- JSON
                    execution_time REAL,
                    error_message TEXT,
                    warnings TEXT,  -- JSON array
                    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (benchmark_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_timestamp 
                ON results (timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_passed 
                ON results (passed)
            """)
    
    def add_source(self, source: ValidationSource) -> None:
        """Add a validation source to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources 
                (source_id, title, authors, year, publisher, isbn, doi, url, 
                 page_reference, table_figure, notes, authority_level, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source.source_id, source.title, source.authors, source.year,
                source.publisher, source.isbn, source.doi, source.url,
                source.page_reference, source.table_figure, source.notes,
                source.authority_level, source.source_type
            ))
    
    def add_benchmark(self, benchmark: ValidationBenchmark) -> None:
        """Add a validation benchmark to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmarks 
                (benchmark_id, source_id, example_name, design_type, outcome_type,
                 calculation_type, test_method, parameters, expected_results, 
                 tolerance, assumptions, notes, verified_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark.benchmark_id, benchmark.source_id, benchmark.example_name,
                benchmark.design_type, benchmark.outcome_type, benchmark.calculation_type,
                benchmark.test_method, json.dumps(benchmark.parameters),
                json.dumps(benchmark.expected_results), benchmark.tolerance,
                json.dumps(benchmark.assumptions or []), benchmark.notes,
                json.dumps(benchmark.verified_by or [])
            ))
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO results 
                (result_id, benchmark_id, timestamp, designpower_version, passed,
                 actual_results, comparisons, execution_time, error_message, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id, result.benchmark_id, result.timestamp,
                result.designpower_version, result.passed,
                json.dumps(result.actual_results), json.dumps(result.comparisons),
                result.execution_time, result.error_message,
                json.dumps(result.warnings or [])
            ))
    
    def get_benchmarks_by_source(self, source_id: str) -> List[ValidationBenchmark]:
        """Get all benchmarks for a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM benchmarks WHERE source_id = ?
            """, (source_id,))
            
            benchmarks = []
            for row in cursor.fetchall():
                benchmark = ValidationBenchmark(
                    benchmark_id=row[0],
                    source_id=row[1],
                    example_name=row[2],
                    design_type=row[3],
                    outcome_type=row[4],
                    calculation_type=row[5],
                    test_method=row[6],
                    parameters=json.loads(row[7]),
                    expected_results=json.loads(row[8]),
                    tolerance=row[9],
                    assumptions=json.loads(row[10]),
                    notes=row[11],
                    verified_by=json.loads(row[12])
                )
                benchmarks.append(benchmark)
            
            return benchmarks
    
    def get_latest_results(self, limit: int = 100) -> List[Tuple[ValidationResult, ValidationBenchmark, ValidationSource]]:
        """Get latest validation results with benchmark and source info."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT r.*, b.example_name, b.design_type, b.outcome_type,
                       s.title, s.authors, s.year, s.page_reference
                FROM results r
                JOIN benchmarks b ON r.benchmark_id = b.benchmark_id
                JOIN sources s ON b.source_id = s.source_id
                ORDER BY r.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = ValidationResult(
                    result_id=row[0],
                    benchmark_id=row[1],
                    timestamp=row[2],
                    designpower_version=row[3],
                    passed=row[4],
                    actual_results=json.loads(row[5]),
                    comparisons=json.loads(row[6]),
                    execution_time=row[7],
                    error_message=row[8],
                    warnings=json.loads(row[9]) if row[9] else []
                )
                
                # Create minimal benchmark and source objects for context
                benchmark = ValidationBenchmark(
                    benchmark_id=row[1],
                    source_id="",
                    example_name=row[10],
                    design_type=row[11],
                    outcome_type=row[12],
                    calculation_type="",
                    test_method="",
                    parameters={},
                    expected_results={},
                    tolerance=0.0,
                    assumptions=[]
                )
                
                source = ValidationSource(
                    source_id="",
                    title=row[13],
                    authors=row[14],
                    year=row[15],
                    publisher="",
                    page_reference=row[16]
                )
                
                results.append((result, benchmark, source))
            
            return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of validation results."""
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_benchmarks,
                    COUNT(DISTINCT source_id) as total_sources,
                    COUNT(DISTINCT design_type) as design_types,
                    COUNT(DISTINCT outcome_type) as outcome_types
                FROM benchmarks
            """)
            overall_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Recent results (last 30 days)
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
                    AVG(execution_time) as avg_execution_time
                FROM results
                WHERE timestamp >= datetime('now', '-30 days')
            """)
            recent_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Pass rates by design type
            cursor = conn.execute("""
                SELECT 
                    b.design_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN r.passed THEN 1 ELSE 0 END) as passed,
                    ROUND(100.0 * SUM(CASE WHEN r.passed THEN 1 ELSE 0 END) / COUNT(*), 1) as pass_rate
                FROM results r
                JOIN benchmarks b ON r.benchmark_id = b.benchmark_id
                WHERE r.timestamp >= datetime('now', '-30 days')
                GROUP BY b.design_type
            """)
            pass_rates = {row[0]: {"total": row[1], "passed": row[2], "pass_rate": row[3]} 
                         for row in cursor.fetchall()}
            
            return {
                "overall_stats": overall_stats,
                "recent_stats": recent_stats,
                "pass_rates_by_design": pass_rates,
                "generated_at": datetime.datetime.now().isoformat()
            }


def generate_validation_id(text: str) -> str:
    """Generate a consistent ID from text."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# Gold standard sources
GOLD_STANDARD_SOURCES = [
    ValidationSource(
        source_id="cohen_1988",
        title="Statistical Power Analysis for the Behavioral Sciences",
        authors="Cohen, Jacob",
        year=1988,
        publisher="Lawrence Erlbaum Associates",
        isbn="9780805802832",
        authority_level="high",
        source_type="textbook",
        notes="Definitive reference for power analysis. Tables widely cited and reproduced."
    ),
    
    ValidationSource(
        source_id="fleiss_2003",
        title="Statistical Methods for Rates and Proportions",
        authors="Fleiss, Joseph L.; Levin, Bruce; Paik, Myunghee Cho",
        year=2003,
        publisher="Wiley",
        isbn="9780471526292",
        authority_level="high",
        source_type="textbook",
        notes="Standard reference for proportion-based sample size calculations."
    ),
    
    ValidationSource(
        source_id="donner_klar_2000",
        title="Design and Analysis of Cluster Randomization Trials in Health Research",
        authors="Donner, Allan; Klar, Neil",
        year=2000,
        publisher="Arnold",
        isbn="9780340691533",
        authority_level="high",
        source_type="textbook",
        notes="Authoritative text on cluster randomized trials."
    ),
    
    ValidationSource(
        source_id="hayes_moulton_2017",
        title="Cluster Randomised Trials, Second Edition",
        authors="Hayes, Richard J.; Moulton, Lawrence H.",
        year=2017,
        publisher="Chapman and Hall/CRC",
        isbn="9781498728225",
        authority_level="high",
        source_type="textbook", 
        notes="Comprehensive modern reference for cluster randomized trials with updated methodologies."
    ),
    
    ValidationSource(
        source_id="lachin_1981",
        title="Introduction to sample size determination and power analysis for clinical trials",
        authors="Lachin, John M.",
        year=1981,
        publisher="Controlled Clinical Trials",
        doi="10.1016/0197-2456(81)90001-5",
        page_reference="Volume 2, Pages 93-113",
        authority_level="high",
        source_type="paper",
        notes="Classic paper on clinical trial sample size methods."
    ),
    
    ValidationSource(
        source_id="r_pwr_package",
        title="R Package 'pwr': Basic Functions for Power Analysis",
        authors="Champely, Stephane",
        year=2020,
        publisher="CRAN",
        url="https://CRAN.R-project.org/package=pwr",
        authority_level="high",
        source_type="software",
        notes="Widely used R package implementing Cohen's methods. De facto standard."
    ),
    
    ValidationSource(
        source_id="sas_proc_power",
        title="SAS/STAT User's Guide: The POWER Procedure",
        authors="SAS Institute Inc.",
        year=2023,
        publisher="SAS Institute",
        url="https://documentation.sas.com/doc/en/statug/15.2/statug_power_toc.htm",
        authority_level="high",
        source_type="software",
        notes="Industrial standard for power analysis in pharmaceutical research."
    ),
    
    ValidationSource(
        source_id="fda_guidance_2010",
        title="Adaptive Design Clinical Trials for Drugs and Biologics Guidance",
        authors="U.S. Food and Drug Administration",
        year=2019,
        publisher="FDA",
        url="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry",
        authority_level="high",
        source_type="regulatory",
        notes="FDA guidance on adaptive designs with sample size considerations."
    ),
    
    ValidationSource(
        source_id="ahern_2001",
        title="Sample size tables for exact single-stage phase II designs",
        authors="A'Hern, Roger P.",
        year=2001,
        publisher="Statistics in Medicine",
        doi="10.1002/sim.721",
        page_reference="Volume 20, Issue 6, Pages 859-866",
        authority_level="high",
        source_type="paper",
        notes="Standard reference for single-arm phase II trial designs."
    ),
    
    ValidationSource(
        source_id="simon_1989",
        title="Optimal two-stage designs for phase II clinical trials",
        authors="Simon, Richard",
        year=1989,
        publisher="Controlled Clinical Trials",
        doi="10.1016/0197-2456(89)90015-9",
        page_reference="Volume 10, Issue 1, Pages 1-10",
        authority_level="high",
        source_type="paper",
        notes="Classic reference for two-stage phase II designs."
    ),
    
    ValidationSource(
        source_id="schoenfeld_1981",
        title="The asymptotic properties of nonparametric tests for comparing survival distributions",
        authors="Schoenfeld, David",
        year=1981,
        publisher="Biometrika",
        doi="10.1093/biomet/68.1.316",
        page_reference="Volume 68, Issue 1, Pages 316-319",
        authority_level="high",
        source_type="paper",
        notes="Fundamental paper for log-rank test sample size calculations."
    ),
    
    ValidationSource(
        source_id="wellek_2010",
        title="Testing Statistical Hypotheses of Equivalence and Noninferiority",
        authors="Wellek, Stefan",
        year=2010,
        publisher="Chapman & Hall/CRC",
        isbn="9781439808184",
        authority_level="high",
        source_type="textbook",
        notes="Definitive reference for non-inferiority and equivalence testing methodology."
    ),
    
    ValidationSource(
        source_id="ich_e9_1998",
        title="Statistical Principles for Clinical Trials - ICH E9",
        authors="International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use",
        year=1998,
        publisher="ICH",
        url="https://database.ich.org/sites/default/files/E9_Guideline.pdf",
        authority_level="high",
        source_type="regulatory",
        notes="International regulatory guidance on statistical principles including non-inferiority trials."
    ),
    
    ValidationSource(
        source_id="fda_ni_guidance_2016",
        title="Non-Inferiority Clinical Trials to Establish Effectiveness - Guidance for Industry",
        authors="U.S. Food and Drug Administration",
        year=2016,
        publisher="FDA",
        url="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials-establish-effectiveness-guidance-industry",
        authority_level="high",
        source_type="regulatory",
        notes="FDA guidance on design and analysis of non-inferiority trials."
    ),
    
    ValidationSource(
        source_id="piaggio_2012",
        title="Reporting of noninferiority and equivalence randomized trials: extension of the CONSORT 2010 statement",
        authors="Piaggio, Gilda; Elbourne, Diana R.; Pocock, Stuart J.; Evans, Stephen J.W.; Altman, Douglas G.",
        year=2012,
        publisher="JAMA",
        doi="10.1001/jama.2012.87802",
        page_reference="Volume 308, Issue 24, Pages 2594-2604",
        authority_level="high",
        source_type="paper",
        notes="CONSORT extension for non-inferiority trials with methodological guidance."
    ),
    
    ValidationSource(
        source_id="chow_liu_2008",
        title="Design and Analysis of Clinical Trials: Concepts and Methodologies",
        authors="Chow, Shein-Chung; Liu, Jen-pei",
        year=2008,
        publisher="Wiley",
        isbn="9780470170526",
        page_reference="Chapter 9: Equivalence and Non-inferiority Trials",
        authority_level="high",
        source_type="textbook",
        notes="Comprehensive coverage of non-inferiority trial methodology with worked examples."
    )
]


def initialize_validation_database(db_path: str = "tests/validation/validation.db") -> ValidationDatabase:
    """Initialize validation database with gold standard sources."""
    db = ValidationDatabase(db_path)
    
    # Add all gold standard sources
    for source in GOLD_STANDARD_SOURCES:
        db.add_source(source)
    
    return db


if __name__ == "__main__":
    # Initialize database with gold standards
    db = initialize_validation_database()
    print("✅ Validation database initialized with gold standard sources")
    
    # Print summary
    summary = db.get_validation_summary()
    print(f"📊 Sources: {summary['overall_stats']['total_sources']}")
    print(f"📋 Benchmarks: {summary['overall_stats']['total_benchmarks']}")
    print(f"🎯 Design types: {summary['overall_stats']['design_types']}")
    print(f"📈 Outcome types: {summary['overall_stats']['outcome_types']}")