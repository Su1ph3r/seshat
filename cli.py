#!/usr/bin/env python3
"""
Seshat CLI - Command-line interface for stylometric analysis.

Usage:
    seshat profile create <name> [--samples <dir>] [--source <source>]
    seshat profile list
    seshat profile show <name> [--verbose]
    seshat profile delete <name>
    seshat analyze <file> [--profiles <dir>] [--detailed]
    seshat compare <file1> <file2> [--detailed]
    seshat psych <file> [--personality] [--emotional] [--cognitive]
    seshat ai-detect <file>
    seshat train [--profiles <dir>] [--output <file>] [--method <method>]
    seshat serve [--host <host>] [--port <port>]
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="seshat")
def main():
    """Seshat - Stylometric Authorship Attribution & Psychological Profiling Tool"""
    pass


@main.group()
def profile():
    """Manage author profiles."""
    pass


@profile.command("create")
@click.argument("name")
@click.option("--samples", "-s", type=click.Path(exists=True), help="Directory containing sample files")
@click.option("--source", default=None, help="Source identifier for samples")
@click.option("--output", "-o", type=click.Path(), help="Output directory for profiles")
def profile_create(name: str, samples: Optional[str], source: Optional[str], output: Optional[str]):
    """Create a new author profile."""
    from seshat.profile import AuthorProfile, ProfileManager
    from seshat.analyzer import Analyzer
    from seshat.io.readers import TextReader

    with console.status(f"Creating profile '{name}'..."):
        if output:
            manager = ProfileManager(storage_dir=output)
        else:
            manager = ProfileManager()

        sample_texts = []
        if samples:
            reader = TextReader()
            try:
                text_samples = reader.read_directory(samples)
                sample_texts = [s.text for s in text_samples]
                console.print(f"Found {len(sample_texts)} samples in {samples}")
            except Exception as e:
                console.print(f"[red]Error reading samples: {e}[/red]")
                return

        try:
            profile = manager.create_profile(
                name=name,
                samples=sample_texts,
                source=source,
            )
            console.print(f"[green]Profile '{name}' created successfully![/green]")
            console.print(f"  Profile ID: {profile.profile_id}")
            console.print(f"  Samples: {profile.get_sample_count()}")
            console.print(f"  Features: {len(profile.aggregated_features)}")
        except Exception as e:
            console.print(f"[red]Error creating profile: {e}[/red]")


@profile.command("list")
@click.option("--output", "-o", type=click.Path(), help="Profile storage directory")
def profile_list(output: Optional[str]):
    """List all profiles."""
    from seshat.profile import ProfileManager

    manager = ProfileManager(storage_dir=output)

    profiles = manager.list_profiles()

    if not profiles:
        console.print("[yellow]No profiles found.[/yellow]")
        return

    table = Table(title="Author Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Created", style="dim")

    for p in profiles:
        table.add_row(
            p["name"],
            str(p["sample_count"]),
            str(p["total_words"]),
            p["created_at"][:10],
        )

    console.print(table)


@profile.command("show")
@click.argument("name")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
@click.option("--output", "-o", type=click.Path(), help="Profile storage directory")
def profile_show(name: str, verbose: bool, output: Optional[str]):
    """Show profile details."""
    from seshat.profile import ProfileManager

    manager = ProfileManager(storage_dir=output)
    profile = manager.get_profile(name)

    if not profile:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        return

    summary = profile.get_summary()

    console.print(Panel(f"[bold]{summary['name']}[/bold]", subtitle=summary['profile_id'][:8]))
    console.print(f"  Created: {summary['created_at']}")
    console.print(f"  Updated: {summary['updated_at']}")
    console.print(f"  Samples: {summary['sample_count']}")
    console.print(f"  Total Words: {summary['total_words']}")
    console.print(f"  Features: {summary['feature_count']}")

    if verbose:
        distinctive = profile.get_distinctive_features(threshold_std=0.5)

        if distinctive:
            console.print("\n[bold]Distinctive Features:[/bold]")
            table = Table()
            table.add_column("Feature")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")

            for feat in distinctive[:15]:
                table.add_row(
                    feat["feature"],
                    f"{feat['mean']:.4f}",
                    f"{feat['std']:.4f}",
                )

            console.print(table)


@profile.command("delete")
@click.argument("name")
@click.option("--output", "-o", type=click.Path(), help="Profile storage directory")
@click.confirmation_option(prompt="Are you sure you want to delete this profile?")
def profile_delete(name: str, output: Optional[str]):
    """Delete a profile."""
    from seshat.profile import ProfileManager

    manager = ProfileManager(storage_dir=output)

    if manager.delete_profile(name):
        console.print(f"[green]Profile '{name}' deleted.[/green]")
    else:
        console.print(f"[red]Profile '{name}' not found.[/red]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--profiles", "-p", type=click.Path(exists=True), help="Profiles directory for comparison")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed analysis")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def analyze(file: str, profiles: Optional[str], detailed: bool, output_json: bool):
    """Analyze a text file for stylometric features."""
    from seshat.analyzer import Analyzer
    from seshat.profile import ProfileManager
    from seshat.comparator import Comparator

    text = Path(file).read_text(encoding="utf-8")

    with console.status("Analyzing text..."):
        analyzer = Analyzer()
        result = analyzer.analyze(text)

    if output_json:
        console.print(result.to_json())
        return

    console.print(Panel(f"[bold]Analysis Results[/bold]", subtitle=file))
    console.print(f"  Words: {result.word_count}")
    console.print(f"  Sentences: {result.sentence_count}")

    if detailed:
        lexical = result.lexical_features
        console.print("\n[bold]Lexical Features:[/bold]")
        console.print(f"  Type-Token Ratio: {lexical.get('type_token_ratio', 0):.4f}")
        console.print(f"  Avg Word Length: {lexical.get('avg_word_length', 0):.2f}")
        console.print(f"  Vocabulary Richness (Yule's K): {lexical.get('yules_k', 0):.4f}")

        punct = result.punctuation_features
        console.print("\n[bold]Punctuation Features:[/bold]")
        console.print(f"  Commas per 1k words: {punct.get('comma_per_1k', 0):.1f}")
        console.print(f"  Question ratio: {punct.get('terminal_question_ratio', 0):.2%}")

    if profiles:
        manager = ProfileManager(storage_dir=profiles)
        profile_list = list(manager.profiles.values())

        if profile_list:
            console.print("\n[bold]Profile Comparison:[/bold]")
            comparator = Comparator(analyzer=analyzer)
            results = comparator.compare_multiple(text, profile_list)

            table = Table()
            table.add_column("Profile")
            table.add_column("Score", justify="right")
            table.add_column("Confidence")

            for r in results[:5]:
                table.add_row(
                    r.profile_name,
                    f"{r.overall_score:.2%}",
                    r.confidence,
                )

            console.print(table)


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option("--detailed", "-d", is_flag=True, help="Show detailed comparison")
def compare(file1: str, file2: str, detailed: bool):
    """Compare two text files for authorship similarity."""
    from seshat.comparator import Comparator

    text1 = Path(file1).read_text(encoding="utf-8")
    text2 = Path(file2).read_text(encoding="utf-8")

    with console.status("Comparing texts..."):
        comparator = Comparator()
        result = comparator.compare_texts(text1, text2)

    console.print(Panel("[bold]Text Comparison[/bold]"))
    console.print(f"  Cosine Similarity: {result['cosine_similarity']:.4f}")
    console.print(f"  Common Features: {result['common_features']}")

    if result['cosine_similarity'] > 0.8:
        assessment = "[green]Likely same author[/green]"
    elif result['cosine_similarity'] > 0.6:
        assessment = "[yellow]Possibly same author[/yellow]"
    else:
        assessment = "[red]Likely different authors[/red]"

    console.print(f"\n  Assessment: {assessment}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--personality", "-p", is_flag=True, help="Show personality analysis")
@click.option("--emotional", "-e", is_flag=True, help="Show emotional analysis")
@click.option("--cognitive", "-c", is_flag=True, help="Show cognitive analysis")
@click.option("--pd-indicators", is_flag=True, help="Show personality disorder indicators (forensic)")
@click.option("--full", "-f", is_flag=True, help="Show all psychological analyses")
def psych(file: str, personality: bool, emotional: bool, cognitive: bool, pd_indicators: bool, full: bool):
    """Perform psychological profiling on text."""
    from seshat.psychology.personality import PersonalityAnalyzer
    from seshat.psychology.emotional import EmotionalAnalyzer
    from seshat.psychology.cognitive import CognitiveAnalyzer
    from seshat.psychology.personality_disorders import PersonalityDisorderIndicators

    text = Path(file).read_text(encoding="utf-8")

    if full:
        personality = emotional = cognitive = pd_indicators = True

    if not (personality or emotional or cognitive or pd_indicators):
        personality = True

    console.print(Panel("[bold]Psychological Profile[/bold]", subtitle=file))

    if personality:
        analyzer = PersonalityAnalyzer()
        result = analyzer.analyze(text)

        console.print("\n[bold]Big Five Personality Traits:[/bold]")

        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        for trait in traits:
            score = result[trait].get("score", 0.5)
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            console.print(f"  {trait.capitalize():20} [{bar}] {score:.2f}")

        if result.get("dominant_traits"):
            console.print(f"\n  Dominant: {', '.join(result['dominant_traits'])}")

    if emotional:
        analyzer = EmotionalAnalyzer()
        result = analyzer.analyze(text)

        console.print("\n[bold]Emotional Tone:[/bold]")
        console.print(f"  Sentiment: {result.get('sentiment_label', 'neutral').capitalize()}")
        console.print(f"  Intensity: {result.get('emotional_intensity', 0):.2f}")
        console.print(f"  Dominant Emotion: {result.get('dominant_emotion', 'neutral').capitalize()}")
        console.print(f"  Authenticity: {result.get('authenticity_score', 0.5):.2f}")

    if cognitive:
        analyzer = CognitiveAnalyzer()
        result = analyzer.analyze(text)

        console.print("\n[bold]Cognitive Style:[/bold]")
        console.print(f"  Analytical Score: {result.get('analytical_score', 0):.2f}")
        console.print(f"  Cognitive Complexity: {result.get('cognitive_complexity', 0):.2f}")
        console.print(f"  Time Orientation: {result.get('time_orientation', 'balanced').capitalize()}")
        console.print(f"  Style: {result.get('cognitive_style', 'balanced').capitalize()}")

    if pd_indicators:
        analyzer = PersonalityDisorderIndicators()
        result = analyzer.analyze(text)

        console.print("\n[bold]Personality Disorder Indicators (Forensic):[/bold]")
        console.print(f"  [dim]{result['disclaimer'][:80]}...[/dim]")

        text_adequacy = result["text_adequacy"]
        confidence_color = "green" if text_adequacy["is_sufficient"] else "yellow"
        console.print(f"\n  Text Adequacy: [{confidence_color}]{text_adequacy['confidence_tier'].replace('_', ' ').title()}[/{confidence_color}] ({text_adequacy['word_count']} words)")

        console.print("\n  [bold]Cluster Scores:[/bold]")
        for cluster_name, cluster_data in result["clusters"].items():
            score = cluster_data["score"]
            bar_len = int(score * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            label = cluster_data["label"]
            console.print(f"    {label:20} [{bar}] {score:.2f}")

        elevated = [(d, data["score"]) for d, data in result["disorders"].items() if data["score"] >= 0.30]
        if elevated:
            elevated.sort(key=lambda x: x[1], reverse=True)
            console.print("\n  [bold]Elevated Markers:[/bold]")
            for disorder, score in elevated[:5]:
                console.print(f"    - {disorder.replace('_', ' ').title()}: {score:.2f}")

        console.print(f"\n  Overall Confidence: {result['confidence'].replace('_', ' ').title()}")


@main.command("pd-analyze")
@click.argument("file", type=click.Path(exists=True))
@click.option("--detailed", "-d", is_flag=True, help="Show detailed dimension breakdowns")
@click.option("--forensic", "-f", is_flag=True, help="Enable forensic mode with metadata")
@click.option("--case-id", default=None, help="Case identifier for forensic tracking")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def pd_analyze(file: str, detailed: bool, forensic: bool, case_id: Optional[str], output_json: bool):
    """Analyze text for personality disorder linguistic indicators (forensic)."""
    import json
    from seshat.psychology.personality_disorders import PersonalityDisorderIndicators

    text = Path(file).read_text(encoding="utf-8")

    with console.status("Analyzing for personality disorder indicators..."):
        analyzer = PersonalityDisorderIndicators()
        if forensic:
            result = analyzer.analyze_forensic(text, case_id=case_id)
        else:
            result = analyzer.analyze(text)

    if output_json:
        console.print(json.dumps(result, indent=2, default=str))
        return

    console.print(Panel("[bold]Personality Disorder Indicator Analysis[/bold]", subtitle=file))

    console.print(f"\n[yellow]{result['disclaimer']}[/yellow]")

    text_adequacy = result["text_adequacy"]
    confidence_color = "green" if text_adequacy["is_sufficient"] else "red"
    console.print(f"\n[bold]Text Adequacy:[/bold]")
    console.print(f"  Word Count: {text_adequacy['word_count']}")
    console.print(f"  Sufficient: [{confidence_color}]{text_adequacy['is_sufficient']}[/{confidence_color}]")
    console.print(f"  Confidence Tier: {text_adequacy['confidence_tier'].replace('_', ' ').title()}")
    if text_adequacy["limitations"]:
        for limitation in text_adequacy["limitations"]:
            console.print(f"  [dim]- {limitation}[/dim]")

    console.print("\n[bold]Cluster Analysis:[/bold]")
    cluster_table = Table()
    cluster_table.add_column("Cluster", style="cyan")
    cluster_table.add_column("Score", justify="right")
    cluster_table.add_column("Interpretation")
    cluster_table.add_column("Disorders")

    for cluster_name, cluster_data in result["clusters"].items():
        cluster_table.add_row(
            cluster_data["label"],
            f"{cluster_data['score']:.3f}",
            cluster_data["interpretation"].title(),
            ", ".join(d.replace("_", " ").title() for d in cluster_data["disorders"]),
        )
    console.print(cluster_table)

    console.print("\n[bold]Individual Disorder Markers:[/bold]")
    disorder_table = Table()
    disorder_table.add_column("Disorder", style="cyan")
    disorder_table.add_column("Score", justify="right")
    disorder_table.add_column("Interpretation")

    sorted_disorders = sorted(
        result["disorders"].items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    for disorder_name, disorder_data in sorted_disorders:
        score = disorder_data["score"]
        if score >= 0.50:
            score_style = "red"
        elif score >= 0.30:
            score_style = "yellow"
        else:
            score_style = "green"

        disorder_table.add_row(
            disorder_name.replace("_", " ").title(),
            f"[{score_style}]{score:.3f}[/{score_style}]",
            disorder_data["interpretation"].title(),
        )
    console.print(disorder_table)

    if detailed:
        console.print("\n[bold]Detailed Dimension Breakdowns:[/bold]")
        for disorder_name, disorder_data in sorted_disorders:
            if disorder_data["score"] >= 0.15:
                console.print(f"\n  [cyan]{disorder_name.replace('_', ' ').title()}:[/cyan]")
                for key, value in disorder_data.items():
                    if key.endswith("_ratio"):
                        dimension = key.replace("_ratio", "").replace("_", " ").title()
                        console.print(f"    {dimension}: {value:.4f}")

    validation = result["validation"]
    console.print("\n[bold]Validation:[/bold]")
    console.print(f"  Feature Coverage: {validation['feature_coverage']:.1%}")
    console.print(f"  Internally Consistent: {'Yes' if validation['is_consistent'] else 'No'}")
    if validation["flags"]:
        console.print("  [yellow]Flags:[/yellow]")
        for flag in validation["flags"]:
            console.print(f"    - {flag}")

    if forensic and "forensic_metadata" in result:
        meta = result["forensic_metadata"]
        console.print("\n[bold]Forensic Metadata:[/bold]")
        console.print(f"  Case ID: {meta['case_id'] or 'Not specified'}")
        console.print(f"  Text Hash (SHA-256): {meta['text_hash'][:16]}...")
        console.print(f"  Analyzed At: {meta['analyzed_at']}")
        console.print(f"  Analyzer Version: {meta['analyzer_version']}")
        if meta["limitations"]:
            console.print("  [yellow]Limitations:[/yellow]")
            for limitation in meta["limitations"]:
                console.print(f"    - {limitation}")

    console.print(f"\n[bold]Overall Confidence:[/bold] {result['confidence'].replace('_', ' ').title()}")


@main.command("ai-detect")
@click.argument("file", type=click.Path(exists=True))
def ai_detect(file: str):
    """Detect if text is AI-generated or human-written."""
    from seshat.advanced.ai_detection import AIDetector

    text = Path(file).read_text(encoding="utf-8")

    with console.status("Analyzing for AI content..."):
        detector = AIDetector()
        result = detector.detect(text)

    console.print(Panel("[bold]AI Detection Results[/bold]", subtitle=file))

    classification = result.get("classification", "unknown")
    ai_prob = result.get("ai_probability", 0.5)

    if "human" in classification:
        color = "green"
    elif "ai" in classification:
        color = "red"
    else:
        color = "yellow"

    console.print(f"  Classification: [{color}]{classification.replace('_', ' ').title()}[/{color}]")
    console.print(f"  AI Probability: {ai_prob:.1%}")
    console.print(f"  Confidence: {result.get('confidence', 'low').capitalize()}")

    if result.get("human_markers"):
        console.print("\n  [green]Human indicators:[/green]")
        for marker in result["human_markers"][:5]:
            console.print(f"    • {marker}")

    if result.get("ai_markers"):
        console.print("\n  [red]AI indicators:[/red]")
        for marker in result["ai_markers"][:5]:
            console.print(f"    • {marker}")


@main.command()
@click.option("--profiles", "-p", type=click.Path(exists=True), required=True, help="Profiles directory")
@click.option("--output", "-o", type=click.Path(), default="model.joblib", help="Output model file")
@click.option("--method", "-m", type=click.Choice(["svm", "random_forest", "ensemble"]), default="svm")
def train(profiles: str, output: str, method: str):
    """Train a classifier from profiles."""
    from seshat.profile import ProfileManager
    from seshat.ml.classifier import AuthorshipClassifier, train_classifier
    from seshat.ml.ensemble import EnsembleClassifier

    manager = ProfileManager(storage_dir=profiles)
    profile_list = list(manager.profiles.values())

    if len(profile_list) < 2:
        console.print("[red]Need at least 2 profiles to train a classifier.[/red]")
        return

    console.print(f"Training {method} classifier with {len(profile_list)} profiles...")

    with console.status("Training..."):
        if method == "ensemble":
            console.print("[yellow]Ensemble training not yet fully implemented.[/yellow]")
            return

        classifier, results = train_classifier(
            profile_list,
            algorithm=method,
            cross_validate=True,
        )

    console.print(f"\n[green]Training complete![/green]")

    if "cross_validation" in results:
        cv = results["cross_validation"]
        console.print(f"  Cross-validation accuracy: {cv['mean_accuracy']:.2%} ± {cv['std_accuracy']:.2%}")

    if "evaluation" in results:
        console.print(f"  Test accuracy: {results['evaluation']['accuracy']:.2%}")

    classifier.save(output)
    console.print(f"  Model saved to: {output}")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--workers", default=1, help="Number of workers")
def serve(host: str, port: int, workers: int):
    """Start the API server."""
    try:
        import uvicorn
        from api.app import app

        console.print(f"Starting Seshat API server on {host}:{port}...")
        uvicorn.run(app, host=host, port=port, workers=workers)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Make sure uvicorn and fastapi are installed.")


if __name__ == "__main__":
    main()
