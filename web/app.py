"""
Seshat Web Application.

Flask-based web interface for stylometric analysis.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_file,
)
from werkzeug.utils import secure_filename

from seshat.analyzer import Analyzer
from seshat.profile import AuthorProfile, ProfileManager
from seshat.comparator import Comparator
from seshat.psychology.personality import PersonalityAnalyzer
from seshat.psychology.emotional import EmotionalAnalyzer
from seshat.psychology.cognitive import CognitiveAnalyzer
from seshat.advanced.ai_detection import AIDetector


UPLOAD_FOLDER = "/tmp/seshat_uploads"
ALLOWED_EXTENSIONS = {"txt", "md", "json", "csv"}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        import secrets
        secret_key = secrets.token_hex(32)
        import warnings
        warnings.warn(
            "SECRET_KEY environment variable not set. Using randomly generated key. "
            "Sessions will not persist across restarts. Set SECRET_KEY for production.",
            UserWarning
        )
    app.secret_key = secret_key
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

    if config:
        app.config.update(config)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    _analyzer = None
    _profile_manager = None

    def get_analyzer():
        nonlocal _analyzer
        if _analyzer is None:
            _analyzer = Analyzer()
        return _analyzer

    def get_profile_manager():
        nonlocal _profile_manager
        if _profile_manager is None:
            storage_dir = app.config.get("PROFILE_STORAGE", "./profiles")
            _profile_manager = ProfileManager(storage_dir=storage_dir)
        return _profile_manager

    @app.route("/")
    def index():
        """Dashboard home page."""
        manager = get_profile_manager()
        profiles = manager.list_profiles()
        return render_template(
            "index.html",
            profiles=profiles,
            profile_count=len(profiles),
        )

    @app.route("/profiles")
    def profiles_list():
        """List all profiles."""
        manager = get_profile_manager()
        profiles = manager.list_profiles()
        return render_template("profiles/list.html", profiles=profiles)

    @app.route("/profiles/create", methods=["GET", "POST"])
    def profile_create():
        """Create a new profile."""
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if not name:
                flash("Profile name is required", "error")
                return render_template("profiles/create.html")

            manager = get_profile_manager()

            if manager.get_profile(name):
                flash(f"Profile '{name}' already exists", "error")
                return render_template("profiles/create.html")

            samples = []
            sample_text = request.form.get("sample_text", "").strip()
            if sample_text:
                samples.append(sample_text)

            if "sample_files" in request.files:
                files = request.files.getlist("sample_files")
                for file in files:
                    if file and allowed_file(file.filename):
                        content = file.read().decode("utf-8", errors="ignore")
                        if content.strip():
                            samples.append(content)

            try:
                profile = manager.create_profile(
                    name=name,
                    samples=samples,
                    source=request.form.get("source"),
                )
                flash(f"Profile '{name}' created successfully", "success")
                return redirect(url_for("profile_view", name=name))
            except Exception as e:
                flash(f"Error creating profile: {e}", "error")
                return render_template("profiles/create.html")

        return render_template("profiles/create.html")

    @app.route("/profiles/<name>")
    def profile_view(name: str):
        """View profile details."""
        manager = get_profile_manager()
        profile = manager.get_profile(name)

        if not profile:
            flash(f"Profile '{name}' not found", "error")
            return redirect(url_for("profiles_list"))

        summary = profile.get_summary()
        distinctive = profile.get_distinctive_features(threshold_std=0.5)[:15]

        return render_template(
            "profiles/view.html",
            profile=profile,
            summary=summary,
            distinctive=distinctive,
        )

    @app.route("/profiles/<name>/delete", methods=["POST"])
    def profile_delete(name: str):
        """Delete a profile."""
        manager = get_profile_manager()

        if manager.delete_profile(name):
            flash(f"Profile '{name}' deleted", "success")
        else:
            flash(f"Profile '{name}' not found", "error")

        return redirect(url_for("profiles_list"))

    @app.route("/profiles/<name>/add-sample", methods=["POST"])
    def profile_add_sample(name: str):
        """Add a sample to a profile."""
        manager = get_profile_manager()
        profile = manager.get_profile(name)

        if not profile:
            flash(f"Profile '{name}' not found", "error")
            return redirect(url_for("profiles_list"))

        sample_text = request.form.get("sample_text", "").strip()

        if not sample_text:
            flash("Sample text is required", "error")
            return redirect(url_for("profile_view", name=name))

        try:
            analyzer = get_analyzer()
            profile.add_sample(
                sample_text,
                source=request.form.get("source"),
                analyzer=analyzer,
            )
            manager.save_profile(profile)
            flash("Sample added successfully", "success")
        except ValueError as e:
            flash(f"Error adding sample: {e}", "error")

        return redirect(url_for("profile_view", name=name))

    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        """Analyze text page."""
        manager = get_profile_manager()
        profiles = manager.list_profiles()

        if request.method == "POST":
            text = request.form.get("text", "").strip()

            if not text:
                if "file" in request.files:
                    file = request.files["file"]
                    if file and allowed_file(file.filename):
                        text = file.read().decode("utf-8", errors="ignore")

            if not text or len(text) < 10:
                flash("Please provide text to analyze (minimum 10 characters)", "error")
                return render_template("analyze.html", profiles=profiles)

            analyzer = get_analyzer()
            result = analyzer.analyze(text)

            compare_profiles = request.form.getlist("profiles")
            comparison_results = None

            if compare_profiles:
                comparator = Comparator(analyzer=analyzer)
                profile_objs = [
                    manager.get_profile(p)
                    for p in compare_profiles
                    if manager.get_profile(p)
                ]
                if profile_objs:
                    comparison_results = comparator.compare_multiple(text, profile_objs)

            return render_template(
                "analyze_results.html",
                text=text[:500] + "..." if len(text) > 500 else text,
                result=result,
                comparison=comparison_results,
                profiles=profiles,
            )

        return render_template("analyze.html", profiles=profiles)

    @app.route("/compare", methods=["GET", "POST"])
    def compare():
        """Compare two texts page."""
        if request.method == "POST":
            text1 = request.form.get("text1", "").strip()
            text2 = request.form.get("text2", "").strip()

            if not text1 or not text2:
                flash("Both texts are required", "error")
                return render_template("compare.html")

            if len(text1) < 10 or len(text2) < 10:
                flash("Each text must be at least 10 characters", "error")
                return render_template("compare.html")

            comparator = Comparator()
            result = comparator.compare_texts(text1, text2)

            return render_template(
                "compare_results.html",
                text1=text1[:300] + "..." if len(text1) > 300 else text1,
                text2=text2[:300] + "..." if len(text2) > 300 else text2,
                result=result,
            )

        return render_template("compare.html")

    @app.route("/psychological", methods=["GET", "POST"])
    def psychological():
        """Psychological profiling page."""
        if request.method == "POST":
            text = request.form.get("text", "").strip()

            if not text or len(text) < 50:
                flash("Please provide text to analyze (minimum 50 characters)", "error")
                return render_template("psychological.html")

            personality_analyzer = PersonalityAnalyzer()
            emotional_analyzer = EmotionalAnalyzer()
            cognitive_analyzer = CognitiveAnalyzer()

            personality = personality_analyzer.analyze(text)
            emotional = emotional_analyzer.analyze(text)
            cognitive = cognitive_analyzer.analyze(text)

            return render_template(
                "psychological_results.html",
                text=text[:500] + "..." if len(text) > 500 else text,
                personality=personality,
                emotional=emotional,
                cognitive=cognitive,
            )

        return render_template("psychological.html")

    @app.route("/ai-detection", methods=["GET", "POST"])
    def ai_detection():
        """AI detection page."""
        if request.method == "POST":
            text = request.form.get("text", "").strip()

            if not text or len(text) < 50:
                flash("Please provide text to analyze (minimum 50 characters)", "error")
                return render_template("ai_detection.html")

            detector = AIDetector()
            result = detector.detect(text)

            return render_template(
                "ai_detection_results.html",
                text=text[:500] + "..." if len(text) > 500 else text,
                result=result,
            )

        return render_template("ai_detection.html")

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        """API endpoint for analysis."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Text is required"}), 400

        text = data["text"]
        if len(text) < 10:
            return jsonify({"error": "Text must be at least 10 characters"}), 400

        analyzer = get_analyzer()
        result = analyzer.analyze(text)

        return jsonify({
            "success": True,
            "analysis": result.to_dict(),
        })

    @app.route("/api/profiles", methods=["GET"])
    def api_profiles():
        """API endpoint to list profiles."""
        manager = get_profile_manager()
        profiles = manager.list_profiles()
        return jsonify({"success": True, "profiles": profiles})

    @app.route("/api/compare", methods=["POST"])
    def api_compare():
        """API endpoint to compare texts."""
        data = request.get_json()
        if not data or "text1" not in data or "text2" not in data:
            return jsonify({"error": "Both text1 and text2 are required"}), 400

        comparator = Comparator()
        result = comparator.compare_texts(data["text1"], data["text2"])

        return jsonify({"success": True, "comparison": result})

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", error="Page not found"), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", error="Internal server error"), 500

    return app


app = create_app()


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=debug_mode, host=host, port=port)
