def lang2text(lang: str) -> str:
    if lang == "zh":
        return "中文"
    elif lang == "en":
        return "English"
    elif lang == "ja":
        return "日本語"
    elif lang == "ko":
        return "한국어"
    elif lang == "de":
        return "Deutsch"
    elif lang == "fr":
        return "Français"
    elif lang == "es":
        return "Español"
    else:
        return lang