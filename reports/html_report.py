"""
HTML Report Generator for Seshat.

Generates interactive HTML reports with visualizations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json


class HTMLReportGenerator:
    """
    Generate HTML reports for stylometric analysis results.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize HTML report generator.

        Args:
            template_dir: Optional directory containing custom templates
        """
        self.template_dir = template_dir

    def generate_analysis_report(
        self,
        analysis_result: Any,
        output_path: str,
        title: str = "Stylometric Analysis Report",
    ) -> str:
        """
        Generate HTML report for analysis results.

        Args:
            analysis_result: AnalysisResult object
            output_path: Path to save HTML file
            title: Report title

        Returns:
            Path to generated report
        """
        result_dict = analysis_result.to_dict()

        html = self._build_analysis_html(result_dict, title)

        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    def generate_comparison_report(
        self,
        comparison_results: List[Any],
        output_path: str,
        title: str = "Authorship Comparison Report",
    ) -> str:
        """
        Generate HTML report for comparison results.

        Args:
            comparison_results: List of comparison result objects
            output_path: Path to save HTML file
            title: Report title

        Returns:
            Path to generated report
        """
        html = self._build_comparison_html(comparison_results, title)

        Path(output_path).write_text(html, encoding="utf-8")
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
        Generate HTML report for psychological analysis.

        Args:
            personality: Big Five personality results
            emotional: Emotional analysis results
            cognitive: Cognitive analysis results
            output_path: Path to save HTML file
            title: Report title

        Returns:
            Path to generated report
        """
        html = self._build_psychological_html(
            personality, emotional, cognitive, title
        )

        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    def _build_analysis_html(
        self, result: Dict[str, Any], title: str
    ) -> str:
        """Build HTML for analysis report."""
        timestamp = datetime.now().isoformat()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        {self._get_report_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="summary">
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat">
                    <span class="value">{result.get('word_count', 0)}</span>
                    <span class="label">Words</span>
                </div>
                <div class="stat">
                    <span class="value">{result.get('sentence_count', 0)}</span>
                    <span class="label">Sentences</span>
                </div>
                <div class="stat">
                    <span class="value">{len(result.get('all_features', {}))}</span>
                    <span class="label">Features</span>
                </div>
            </div>
        </section>

        <section class="features">
            <h2>Lexical Features</h2>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
                {self._features_to_rows(result.get('lexical_features', {}))}
            </table>
        </section>

        <section class="features">
            <h2>Function Word Features</h2>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
                {self._features_to_rows(result.get('function_word_features', {}))}
            </table>
        </section>

        <section class="features">
            <h2>Punctuation Features</h2>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
                {self._features_to_rows(result.get('punctuation_features', {}))}
            </table>
        </section>

        <section class="features">
            <h2>Syntactic Features</h2>
            <table>
                <tr><th>Feature</th><th>Value</th></tr>
                {self._features_to_rows(result.get('syntactic_features', {}))}
            </table>
        </section>

        <section class="all-features">
            <h2>All Features</h2>
            <details>
                <summary>View all {len(result.get('all_features', {}))} features</summary>
                <table>
                    <tr><th>Feature</th><th>Value</th></tr>
                    {self._features_to_rows(result.get('all_features', {}))}
                </table>
            </details>
        </section>

        <footer>
            <p>Generated by Seshat - Stylometric Authorship Attribution & Psychological Profiling</p>
        </footer>
    </div>
</body>
</html>"""

    def _build_comparison_html(
        self, results: List[Any], title: str
    ) -> str:
        """Build HTML for comparison report."""
        timestamp = datetime.now().isoformat()

        results_html = ""
        for i, r in enumerate(results[:10], 1):
            score_pct = r.overall_score * 100 if hasattr(r, 'overall_score') else 0
            conf = r.confidence if hasattr(r, 'confidence') else 'unknown'
            name = r.profile_name if hasattr(r, 'profile_name') else f'Profile {i}'

            results_html += f"""
            <tr class="{'highlight' if i == 1 else ''}">
                <td>{i}</td>
                <td>{name}</td>
                <td>
                    <div class="score-bar">
                        <div class="fill" style="width: {score_pct}%"></div>
                        <span>{score_pct:.1f}%</span>
                    </div>
                </td>
                <td class="confidence-{conf}">{conf}</td>
            </tr>"""

        best_match = results[0] if results else None
        best_name = best_match.profile_name if best_match and hasattr(best_match, 'profile_name') else 'None'
        best_score = best_match.overall_score * 100 if best_match and hasattr(best_match, 'overall_score') else 0

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        {self._get_report_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="best-match">
            <h2>Best Match</h2>
            <div class="match-result">
                <span class="profile-name">{best_name}</span>
                <span class="score">{best_score:.1f}%</span>
            </div>
        </section>

        <section class="comparison-results">
            <h2>All Candidates</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Profile</th>
                    <th>Score</th>
                    <th>Confidence</th>
                </tr>
                {results_html}
            </table>
        </section>

        <div id="comparison-chart"></div>

        <footer>
            <p>Generated by Seshat - Stylometric Authorship Attribution & Psychological Profiling</p>
        </footer>
    </div>

    <script>
        var names = {json.dumps([r.profile_name if hasattr(r, 'profile_name') else f'Profile {i}' for i, r in enumerate(results[:10], 1)])};
        var scores = {json.dumps([r.overall_score * 100 if hasattr(r, 'overall_score') else 0 for r in results[:10]])};

        Plotly.newPlot('comparison-chart', [{{
            type: 'bar',
            x: names,
            y: scores,
            marker: {{ color: scores.map(s => s > 80 ? '#28a745' : s > 60 ? '#ffc107' : '#dc3545') }}
        }}], {{
            title: 'Profile Similarity Scores',
            yaxis: {{ title: 'Similarity %', range: [0, 100] }}
        }});
    </script>
</body>
</html>"""

    def _build_psychological_html(
        self,
        personality: Dict[str, Any],
        emotional: Dict[str, Any],
        cognitive: Dict[str, Any],
        title: str,
    ) -> str:
        """Build HTML for psychological report."""
        timestamp = datetime.now().isoformat()

        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        trait_scores = [personality.get(t, {}).get('score', 0.5) for t in traits]

        trait_rows = ""
        for trait in traits:
            data = personality.get(trait, {})
            score = data.get('score', 0.5)
            trait_rows += f"""
            <tr>
                <td>{trait.capitalize()}</td>
                <td>
                    <div class="trait-bar">
                        <div class="fill" style="width: {score * 100}%"></div>
                    </div>
                </td>
                <td>{score:.2f}</td>
                <td>{'High' if score >= 0.7 else 'Moderate' if score >= 0.4 else 'Low'}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        {self._get_report_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="personality">
            <h2>Big Five Personality Traits</h2>
            <table>
                <tr>
                    <th>Trait</th>
                    <th>Score</th>
                    <th>Value</th>
                    <th>Level</th>
                </tr>
                {trait_rows}
            </table>
            <div id="personality-chart"></div>
        </section>

        <section class="emotional">
            <h2>Emotional Tone</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Sentiment</td><td>{emotional.get('sentiment_label', 'Neutral').capitalize()}</td></tr>
                <tr><td>Sentiment Score</td><td>{emotional.get('sentiment_score', 0):.2f}</td></tr>
                <tr><td>Emotional Intensity</td><td>{emotional.get('emotional_intensity', 0):.2f}</td></tr>
                <tr><td>Dominant Emotion</td><td>{emotional.get('dominant_emotion', 'Neutral').capitalize()}</td></tr>
                <tr><td>Authenticity</td><td>{emotional.get('authenticity_score', 0.5):.2f}</td></tr>
            </table>
        </section>

        <section class="cognitive">
            <h2>Cognitive Style</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Analytical Score</td><td>{cognitive.get('analytical_score', 0):.2f}</td></tr>
                <tr><td>Cognitive Complexity</td><td>{cognitive.get('cognitive_complexity', 0):.2f}</td></tr>
                <tr><td>Time Orientation</td><td>{cognitive.get('time_orientation', 'Balanced').capitalize()}</td></tr>
                <tr><td>Cognitive Style</td><td>{cognitive.get('cognitive_style', 'Balanced').capitalize()}</td></tr>
            </table>
        </section>

        <footer>
            <p>Generated by Seshat - Stylometric Authorship Attribution & Psychological Profiling</p>
            <p class="disclaimer">Note: These are probabilistic estimates based on linguistic correlations, not clinical diagnoses.</p>
        </footer>
    </div>

    <script>
        var traits = {json.dumps([t.capitalize() for t in traits])};
        var scores = {json.dumps(trait_scores)};

        Plotly.newPlot('personality-chart', [{{
            type: 'scatterpolar',
            r: scores.concat([scores[0]]),
            theta: traits.concat([traits[0]]),
            fill: 'toself',
            fillcolor: 'rgba(74, 144, 217, 0.3)',
            line: {{ color: 'rgba(74, 144, 217, 1)' }}
        }}], {{
            polar: {{
                radialaxis: {{
                    visible: true,
                    range: [0, 1]
                }}
            }},
            title: 'Big Five Personality Profile',
            showlegend: false
        }});
    </script>
</body>
</html>"""

    def _features_to_rows(self, features: Dict[str, Any]) -> str:
        """Convert features dict to HTML table rows."""
        rows = []
        for key, value in features.items():
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            rows.append(f"<tr><td>{key}</td><td>{formatted}</td></tr>")
        return "\n".join(rows)

    def _get_report_css(self) -> str:
        """Return CSS for reports."""
        return """
        :root {
            --primary: #4a90d9;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--primary);
        }

        header h1 { color: var(--primary); }
        .timestamp { color: #666; font-size: 0.9rem; }

        section {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        h2 {
            color: var(--primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }

        .stats {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }

        .stat .value {
            display: block;
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary);
        }

        .stat .label {
            color: #666;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
        }

        tr:hover { background: #f8f9fa; }
        tr.highlight { background: #e8f4fd; }

        .score-bar, .trait-bar {
            position: relative;
            height: 24px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }

        .score-bar .fill, .trait-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
        }

        .score-bar span {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: bold;
            font-size: 0.85rem;
        }

        .confidence-high { color: var(--success); font-weight: bold; }
        .confidence-medium { color: var(--warning); font-weight: bold; }
        .confidence-low { color: var(--danger); font-weight: bold; }

        .best-match { text-align: center; }
        .match-result { margin: 1rem 0; }
        .profile-name { font-size: 1.5rem; font-weight: bold; display: block; }
        .score { font-size: 3rem; font-weight: bold; color: var(--primary); }

        details { margin-top: 1rem; }
        summary { cursor: pointer; color: var(--primary); font-weight: bold; }

        footer {
            text-align: center;
            color: #666;
            padding: 2rem;
            font-size: 0.9rem;
        }

        .disclaimer {
            font-style: italic;
            margin-top: 0.5rem;
        }

        @media print {
            body { background: white; }
            .container { max-width: none; padding: 0; }
            section { box-shadow: none; border: 1px solid #ddd; }
        }
        """
