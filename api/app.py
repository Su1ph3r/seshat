"""
FastAPI application for Seshat REST API.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Seshat API",
    description="Stylometric Authorship Attribution & Psychological Profiling API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    """Input model for text analysis."""
    text: str = Field(..., min_length=10, description="Text to analyze")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class ProfileCreate(BaseModel):
    """Input model for profile creation."""
    name: str = Field(..., min_length=1, description="Profile name")
    samples: List[str] = Field(default=[], description="Initial text samples")
    source: Optional[str] = Field(default=None, description="Source identifier")


class ProfileSample(BaseModel):
    """Input model for adding samples to a profile."""
    text: str = Field(..., min_length=10, description="Sample text")
    source: Optional[str] = Field(default=None, description="Source identifier")


class CompareInput(BaseModel):
    """Input model for text comparison."""
    text1: str = Field(..., min_length=10, description="First text")
    text2: str = Field(..., min_length=10, description="Second text")


_profiles: Dict[str, Any] = {}
_analyzer = None


def get_analyzer():
    """Get or create analyzer instance."""
    global _analyzer
    if _analyzer is None:
        from seshat.analyzer import Analyzer
        _analyzer = Analyzer()
    return _analyzer


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Seshat API",
        "version": "0.1.0",
        "description": "Stylometric Authorship Attribution & Psychological Profiling",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/analyze")
async def analyze_text(input_data: TextInput):
    """
    Analyze text for stylometric features.

    Returns comprehensive stylometric analysis including lexical,
    syntactic, and psychological features.
    """
    analyzer = get_analyzer()

    try:
        result = analyzer.analyze(input_data.text, metadata=input_data.metadata)
        return {
            "success": True,
            "analysis": result.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/quick")
async def quick_analyze(input_data: TextInput):
    """
    Perform quick analysis with minimal features.

    Returns key metrics for rapid assessment.
    """
    from seshat.analyzer import QuickAnalyzer

    analyzer = QuickAnalyzer()

    try:
        result = analyzer.analyze(input_data.text)
        return {
            "success": True,
            "analysis": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/psychological")
async def psychological_analysis(input_data: TextInput):
    """
    Perform psychological profiling on text.

    Returns Big Five personality traits, emotional tone,
    and cognitive style analysis.
    """
    from seshat.psychology.personality import PersonalityAnalyzer
    from seshat.psychology.emotional import EmotionalAnalyzer
    from seshat.psychology.cognitive import CognitiveAnalyzer

    try:
        personality = PersonalityAnalyzer().analyze(input_data.text)
        emotional = EmotionalAnalyzer().analyze(input_data.text)
        cognitive = CognitiveAnalyzer().analyze(input_data.text)

        return {
            "success": True,
            "personality": personality,
            "emotional": emotional,
            "cognitive": cognitive,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/ai-detection")
async def ai_detection(input_data: TextInput):
    """
    Detect if text is AI-generated or human-written.
    """
    from seshat.advanced.ai_detection import AIDetector

    try:
        detector = AIDetector()
        result = detector.detect(input_data.text)
        return {
            "success": True,
            "detection": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compare")
async def compare_texts(input_data: CompareInput):
    """
    Compare two texts for authorship similarity.
    """
    from seshat.comparator import Comparator

    try:
        comparator = Comparator()
        result = comparator.compare_texts(input_data.text1, input_data.text2)
        return {
            "success": True,
            "comparison": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PersonalityDisorderInput(BaseModel):
    """Input model for personality disorder indicator analysis."""
    text: str = Field(..., min_length=10, description="Text to analyze")
    forensic_mode: bool = Field(default=False, description="Enable forensic mode with metadata")
    case_id: Optional[str] = Field(default=None, description="Case identifier for forensic tracking")


@app.post("/api/v1/analyze/personality-disorder-indicators")
async def personality_disorder_indicators(input_data: PersonalityDisorderInput):
    """
    Analyze text for personality disorder linguistic indicators.

    CRITICAL: These are linguistic correlations, NOT clinical diagnoses.
    For forensic and research use by qualified professionals only.

    Returns analysis of linguistic markers associated with DSM-5 personality
    disorder clusters (A: Odd/Eccentric, B: Dramatic/Emotional, C: Anxious/Fearful).
    """
    from seshat.psychology.personality_disorders import PersonalityDisorderIndicators

    try:
        analyzer = PersonalityDisorderIndicators()

        if input_data.forensic_mode:
            result = analyzer.analyze_forensic(
                input_data.text,
                case_id=input_data.case_id,
            )
        else:
            result = analyzer.analyze(input_data.text)

        return {
            "success": True,
            "analysis": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/profiles")
async def create_profile(profile_data: ProfileCreate):
    """
    Create a new author profile.
    """
    from seshat.profile import AuthorProfile

    if profile_data.name in _profiles:
        raise HTTPException(status_code=400, detail="Profile already exists")

    try:
        profile = AuthorProfile.create(name=profile_data.name)
        analyzer = get_analyzer()

        for sample in profile_data.samples:
            try:
                profile.add_sample(
                    sample,
                    source=profile_data.source,
                    analyzer=analyzer,
                )
            except ValueError:
                continue

        _profiles[profile_data.name] = profile

        return {
            "success": True,
            "profile": profile.get_summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/profiles")
async def list_profiles():
    """
    List all profiles.
    """
    return {
        "success": True,
        "profiles": [p.get_summary() for p in _profiles.values()],
    }


@app.get("/api/v1/profiles/{name}")
async def get_profile(name: str):
    """
    Get a specific profile.
    """
    if name not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = _profiles[name]

    return {
        "success": True,
        "profile": profile.to_dict(),
    }


@app.post("/api/v1/profiles/{name}/samples")
async def add_sample(name: str, sample_data: ProfileSample):
    """
    Add a sample to an existing profile.
    """
    if name not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        profile = _profiles[name]
        analyzer = get_analyzer()

        profile.add_sample(
            sample_data.text,
            source=sample_data.source,
            analyzer=analyzer,
        )

        return {
            "success": True,
            "profile": profile.get_summary(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/profiles/{name}")
async def delete_profile(name: str):
    """
    Delete a profile.
    """
    if name not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    del _profiles[name]

    return {
        "success": True,
        "message": f"Profile '{name}' deleted",
    }


@app.post("/api/v1/profiles/{name}/compare")
async def compare_to_profile(name: str, input_data: TextInput):
    """
    Compare text against a specific profile.
    """
    from seshat.comparator import Comparator

    if name not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        profile = _profiles[name]
        comparator = Comparator(analyzer=get_analyzer())

        result = comparator.compare(input_data.text, profile)

        return {
            "success": True,
            "comparison": result.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/attribute")
async def attribute_text(input_data: TextInput):
    """
    Attribute text to the most likely author from all profiles.
    """
    from seshat.comparator import Comparator

    if not _profiles:
        raise HTTPException(status_code=400, detail="No profiles available")

    try:
        comparator = Comparator(analyzer=get_analyzer())
        results = comparator.compare_multiple(input_data.text, list(_profiles.values()))

        return {
            "success": True,
            "attribution": {
                "best_match": results[0].profile_name if results else None,
                "confidence": results[0].confidence if results else "None",
                "score": results[0].overall_score if results else 0,
                "candidates": [
                    {
                        "profile": r.profile_name,
                        "score": r.overall_score,
                        "confidence": r.confidence,
                    }
                    for r in results[:5]
                ],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
