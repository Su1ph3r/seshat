"""
PDF Report Generator for Seshat.

Generates PDF reports using WeasyPrint or falls back to HTML.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from reports.html_report import HTMLReportGenerator


class PDFReportGenerator:
    """
    Generate PDF reports for stylometric analysis results.

    Uses WeasyPrint if available, otherwise generates HTML.
    """

    def __init__(self):
        """Initialize PDF report generator."""
        self.html_generator = HTMLReportGenerator()
        self._weasyprint_available = self._check_weasyprint()

    def _check_weasyprint(self) -> bool:
        """Check if WeasyPrint is available."""
        try:
            import weasyprint  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_analysis_report(
        self,
        analysis_result: Any,
        output_path: str,
        title: str = "Stylometric Analysis Report",
    ) -> str:
        """
        Generate PDF report for analysis results.

        Args:
            analysis_result: AnalysisResult object
            output_path: Path to save PDF file
            title: Report title

        Returns:
            Path to generated report
        """
        if not self._weasyprint_available:
            html_path = output_path.replace(".pdf", ".html")
            return self.html_generator.generate_analysis_report(
                analysis_result, html_path, title
            )

        from weasyprint import HTML

        result_dict = analysis_result.to_dict()
        html_content = self.html_generator._build_analysis_html(result_dict, title)

        HTML(string=html_content).write_pdf(output_path)
        return output_path

    def generate_comparison_report(
        self,
        comparison_results: List[Any],
        output_path: str,
        title: str = "Authorship Comparison Report",
    ) -> str:
        """
        Generate PDF report for comparison results.

        Args:
            comparison_results: List of comparison result objects
            output_path: Path to save PDF file
            title: Report title

        Returns:
            Path to generated report
        """
        if not self._weasyprint_available:
            html_path = output_path.replace(".pdf", ".html")
            return self.html_generator.generate_comparison_report(
                comparison_results, html_path, title
            )

        from weasyprint import HTML

        html_content = self.html_generator._build_comparison_html(
            comparison_results, title
        )

        HTML(string=html_content).write_pdf(output_path)
        return output_path

    def generate_psychological_report(
        self,
        personality: Dict[str, Any],
        emotional: Dict[str, Any],
        cognitive: Dict[str, Any],
        output_path: str,
        title: str = "Psychological Profile Report",
    ) -> str:
        """
        Generate PDF report for psychological analysis.

        Args:
            personality: Big Five personality results
            emotional: Emotional analysis results
            cognitive: Cognitive analysis results
            output_path: Path to save PDF file
            title: Report title

        Returns:
            Path to generated report
        """
        if not self._weasyprint_available:
            html_path = output_path.replace(".pdf", ".html")
            return self.html_generator.generate_psychological_report(
                personality, emotional, cognitive, html_path, title
            )

        from weasyprint import HTML

        html_content = self.html_generator._build_psychological_html(
            personality, emotional, cognitive, title
        )

        HTML(string=html_content).write_pdf(output_path)
        return output_path

    def generate_forensic_report(
        self,
        analysis_result: Any,
        comparison_results: Optional[List[Any]],
        evidence_chain: Optional[Dict[str, Any]],
        output_path: str,
        title: str = "Forensic Analysis Report",
    ) -> str:
        """
        Generate comprehensive forensic report.

        Args:
            analysis_result: AnalysisResult object
            comparison_results: Optional comparison results
            evidence_chain: Optional evidence chain data
            output_path: Path to save report
            title: Report title

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().isoformat()
        result_dict = analysis_result.to_dict() if analysis_result else {}

        html_content = self._build_forensic_html(
            result_dict, comparison_results, evidence_chain, title, timestamp
        )

        if self._weasyprint_available:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(output_path)
        else:
            html_path = output_path.replace(".pdf", ".html")
            Path(html_path).write_text(html_content, encoding="utf-8")
            output_path = html_path

        return output_path

    def _build_forensic_html(
        self,
        result: Dict[str, Any],
        comparison: Optional[List[Any]],
        evidence: Optional[Dict[str, Any]],
        title: str,
        timestamp: str,
    ) -> str:
        """Build HTML for forensic report."""
        evidence_section = ""
        if evidence:
            evidence_section = f"""
            <section class="evidence">
                <h2>Evidence Chain</h2>
                <table>
                    <tr><th>Field</th><th>Value</th></tr>
                    <tr><td>Evidence ID</td><td>{evidence.get('id', 'N/A')}</td></tr>
                    <tr><td>Content Hash (SHA-256)</td><td><code>{evidence.get('content_hash', 'N/A')}</code></td></tr>
                    <tr><td>Collected At</td><td>{evidence.get('collected_at', 'N/A')}</td></tr>
                    <tr><td>Collected By</td><td>{evidence.get('collected_by', 'N/A')}</td></tr>
                    <tr><td>Source URL</td><td>{evidence.get('source_url', 'N/A')}</td></tr>
                    <tr><td>Collection Method</td><td>{evidence.get('collection_method', 'N/A')}</td></tr>
                </table>
            </section>
            """

        comparison_section = ""
        if comparison:
            rows = ""
            for i, r in enumerate(comparison[:10], 1):
                score_pct = r.overall_score * 100 if hasattr(r, 'overall_score') else 0
                conf = r.confidence if hasattr(r, 'confidence') else 'unknown'
                name = r.profile_name if hasattr(r, 'profile_name') else f'Profile {i}'
                rows += f"<tr><td>{i}</td><td>{name}</td><td>{score_pct:.2f}%</td><td>{conf}</td></tr>"

            comparison_section = f"""
            <section class="comparison">
                <h2>Authorship Comparison</h2>
                <table>
                    <tr><th>Rank</th><th>Profile</th><th>Score</th><th>Confidence</th></tr>
                    {rows}
                </table>
            </section>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {self.html_generator._get_report_css()}
        .forensic-header {{
            background: #2c3e50;
            color: white;
            padding: 2rem;
            margin: -2rem -2rem 2rem -2rem;
            text-align: center;
        }}
        .forensic-header h1 {{ color: white; }}
        .methodology {{
            background: #f8f9fa;
            border-left: 4px solid #4a90d9;
            padding: 1rem;
            margin: 1rem 0;
        }}
        code {{
            font-family: monospace;
            background: #f1f1f1;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.85rem;
            word-break: break-all;
        }}
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="forensic-header">
            <h1>{title}</h1>
            <p>Generated: {timestamp}</p>
            <p>CONFIDENTIAL - FOR AUTHORIZED USE ONLY</p>
        </div>

        <section class="methodology">
            <h2>Methodology</h2>
            <p>This report was generated using Seshat stylometric analysis software.
            The analysis employs established computational linguistics methods including:</p>
            <ul>
                <li>Lexical feature extraction (vocabulary richness, word length distribution)</li>
                <li>Function word analysis (pronouns, articles, prepositions)</li>
                <li>Punctuation pattern analysis</li>
                <li>Character and word n-gram analysis</li>
                <li>Syntactic complexity metrics</li>
            </ul>
            <p>Similarity scores are computed using cosine similarity and Burrow's Delta
            on normalized feature vectors. Confidence levels are based on the margin
            between top candidates and cross-validation consistency.</p>
        </section>

        {evidence_section}

        <section class="analysis">
            <h2>Stylometric Analysis</h2>
            <h3>Summary</h3>
            <table>
                <tr><td>Word Count</td><td>{result.get('word_count', 0)}</td></tr>
                <tr><td>Sentence Count</td><td>{result.get('sentence_count', 0)}</td></tr>
                <tr><td>Features Extracted</td><td>{len(result.get('all_features', {}))}</td></tr>
            </table>

            <h3>Key Features</h3>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
                <tr><td>Type-Token Ratio</td><td>{result.get('lexical_features', {}).get('type_token_ratio', 0):.4f}</td></tr>
                <tr><td>Avg Word Length</td><td>{result.get('lexical_features', {}).get('avg_word_length', 0):.2f}</td></tr>
                <tr><td>Avg Sentence Length</td><td>{result.get('syntactic_features', {}).get('avg_sentence_length_words', 0):.1f}</td></tr>
                <tr><td>First Person Singular Ratio</td><td>{result.get('function_word_features', {}).get('first_person_singular_ratio', 0):.4f}</td></tr>
            </table>
        </section>

        {comparison_section}

        <section class="limitations">
            <h2>Limitations and Caveats</h2>
            <div class="disclaimer">
                <p><strong>Important:</strong> Stylometric analysis provides probabilistic
                assessments based on statistical patterns. Results should be considered
                as one piece of evidence among many and should not be used as the sole
                basis for definitive conclusions.</p>
                <ul>
                    <li>Accuracy depends on the quantity and quality of reference samples</li>
                    <li>Authors may intentionally disguise their writing style</li>
                    <li>Writing style can vary across genres, platforms, and time</li>
                    <li>Machine translation or editing may affect stylometric signatures</li>
                </ul>
            </div>
        </section>

        <footer>
            <p>Generated by Seshat - Stylometric Authorship Attribution & Psychological Profiling</p>
            <p>This report should be interpreted by qualified analysts.</p>
        </footer>
    </div>
</body>
</html>"""
