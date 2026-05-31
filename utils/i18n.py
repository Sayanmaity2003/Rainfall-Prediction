"""Small translation helpers for the app UI."""

TRANSLATIONS = {
    "English": {
        "home": "Home",
        "manual": "Manual Prediction",
        "live": "Live Location",
        "analytics": "AI Analytics",
        "hero_title": "Rainfall Detection using Machine Learning",
        "hero_subtitle": "Rainfall Prediction System",
        "predict": "Predict Rainfall",
        "rain": "Rainfall Expected",
        "no_rain": "No Rainfall Expected",
        "confidence": "Confidence",
        "risk": "Flood Risk",
        "recommendation": "Recommendation",
        "summary": "AI Weather Summary",
    },
    "Hindi": {
        "home": "होम",
        "manual": "मैनुअल पूर्वानुमान",
        "live": "लाइव लोकेशन",
        "analytics": "एआई एनालिटिक्स",
        "hero_title": "मशीन लर्निंग द्वारा वर्षा पहचान",
        "hero_subtitle": "एआई-संचालित वर्षा पूर्वानुमान और जलवायु इंटेलिजेंस सिस्टम",
        "predict": "वर्षा का अनुमान लगाएं",
        "rain": "वर्षा की संभावना है",
        "no_rain": "वर्षा की संभावना नहीं",
        "confidence": "विश्वास स्तर",
        "risk": "बाढ़ जोखिम",
        "recommendation": "सुझाव",
        "summary": "एआई मौसम सारांश",
    },
    "Bengali": {
        "home": "হোম",
        "manual": "ম্যানুয়াল পূর্বাভাস",
        "live": "লাইভ লোকেশন",
        "analytics": "এআই অ্যানালিটিক্স",
        "hero_title": "মেশিন লার্নিং দিয়ে বৃষ্টিপাত শনাক্তকরণ",
        "hero_subtitle": "এআই-চালিত বৃষ্টিপাত পূর্বাভাস ও জলবায়ু ইন্টেলিজেন্স সিস্টেম",
        "predict": "বৃষ্টিপাত পূর্বাভাস করুন",
        "rain": "বৃষ্টিপাতের সম্ভাবনা আছে",
        "no_rain": "বৃষ্টিপাতের সম্ভাবনা নেই",
        "confidence": "বিশ্বাসযোগ্যতা",
        "risk": "বন্যার ঝুঁকি",
        "recommendation": "পরামর্শ",
        "summary": "এআই আবহাওয়া সারাংশ",
    },
}


def t(language: str, key: str) -> str:
    """Return translated text with English fallback."""
    return TRANSLATIONS.get(language, TRANSLATIONS["English"]).get(
        key, TRANSLATIONS["English"].get(key, key)
    )

