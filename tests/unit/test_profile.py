"""
Unit tests for profile management.
"""

import pytest
from seshat.profile import AuthorProfile, ProfileManager, Sample


class TestAuthorProfile:
    """Tests for AuthorProfile class."""

    def test_profile_create(self):
        profile = AuthorProfile.create(name="TestAuthor")
        assert profile.name == "TestAuthor"
        assert profile.profile_id is not None
        assert len(profile.samples) == 0

    def test_profile_add_sample(self, analyzer):
        profile = AuthorProfile.create(name="TestAuthor")
        text = "This is a test sample with enough words to be valid."
        profile.add_sample(text, analyzer=analyzer)
        assert len(profile.samples) == 1
        assert profile.samples[0].text == text

    def test_profile_add_short_sample(self, analyzer):
        profile = AuthorProfile.create(name="TestAuthor")
        with pytest.raises(ValueError):
            profile.add_sample("short", analyzer=analyzer)

    def test_profile_get_summary(self, sample_profile):
        summary = sample_profile.get_summary()
        assert "name" in summary
        assert "sample_count" in summary
        assert "total_words" in summary

    def test_profile_aggregated_features(self, sample_profile):
        assert len(sample_profile.aggregated_features) > 0

    def test_profile_to_dict(self, sample_profile):
        profile_dict = sample_profile.to_dict()
        assert isinstance(profile_dict, dict)
        assert "name" in profile_dict
        assert "samples" in profile_dict


class TestProfileManager:
    """Tests for ProfileManager class."""

    def test_manager_init(self, tmp_path):
        manager = ProfileManager(storage_dir=str(tmp_path / "profiles"))
        assert manager is not None

    def test_manager_create_profile(self, profile_manager):
        profile = profile_manager.create_profile(
            name="TestAuthor",
            samples=["This is a test sample with enough words."],
        )
        assert profile.name == "TestAuthor"

    def test_manager_get_profile(self, profile_manager):
        profile_manager.create_profile(
            name="TestAuthor",
            samples=["This is a test sample with enough words."],
        )
        retrieved = profile_manager.get_profile("TestAuthor")
        assert retrieved is not None
        assert retrieved.name == "TestAuthor"

    def test_manager_list_profiles(self, profile_manager):
        profile_manager.create_profile(name="Author1", samples=["Sample text one."])
        profile_manager.create_profile(name="Author2", samples=["Sample text two."])
        profiles = profile_manager.list_profiles()
        assert len(profiles) >= 2

    def test_manager_delete_profile(self, profile_manager):
        profile_manager.create_profile(name="ToDelete", samples=["Sample text."])
        result = profile_manager.delete_profile("ToDelete")
        assert result is True
        assert profile_manager.get_profile("ToDelete") is None


class TestSample:
    """Tests for Sample class."""

    def test_sample_creation(self):
        sample = Sample(
            text="This is a test sample.",
            word_count=5,
            features={"test": 1.0},
        )
        assert sample.text == "This is a test sample."
        assert sample.word_count == 5

    def test_sample_to_dict(self):
        sample = Sample(
            text="Test text",
            word_count=2,
            features={"feature1": 0.5},
        )
        sample_dict = sample.to_dict()
        assert "text" in sample_dict
        assert "features" in sample_dict
