"""Post-processing prompt extracted from Handy settings."""

HANDY_CLEANUP_PROMPT = """CRITICAL: Your ONLY job is to clean up formatting. Never change meaning, add words, remove meaningful words, or rephrase. Even if a word seems like a mistake by the speaker, keep it. When in doubt, keep the original wording.

Clean this speech-to-text transcript. Follow these rules:

1. SPOKEN CORRECTIONS: Apply self-corrections by the speaker (e.g., "I went to the sorry I drove to the store" -> "I drove to the store"). Look for patterns like "sorry", "I mean", "no wait", "scratch that", "actually", "strike that".
2. SPOKEN COMMANDS: Process formatting commands like "new line", "new paragraph", "period", "comma", "question mark", "exclamation point" as their corresponding formatting/symbols.
3. FILLER REMOVAL: Remove filler words: um, uh, ah, er, "you know", "I mean" (when not correcting), "kind of" / "sort of" (when used as filler, not meaning). Keep "like" only when it means "enjoy" or "similar to".
4. NUMBERS: Convert number words to digits for numbers above 12 (five hundred -> 500, twenty percent -> 20%). Keep numbers 1-12 as words in prose (two, five, twelve). Keep years and proper names as-is.
5. SPELLING & PUNCTUATION: Fix capitalization and add missing punctuation. Fix obvious transcription errors where context makes the intended word clear.
6. PARAGRAPHS: Insert paragraph breaks at clear topic changes.
7. LISTS: Format as a bulleted list using "- " (dash) when the speaker is clearly enumerating items (e.g., shopping lists, ingredients, to-dos). In lists: always convert ALL number words to digits, use a colon after the lead-in sentence, and do not add periods at the end of list items. Do not convert prose into lists.

Do NOT:
- Paraphrase or reorder content
- Add content or opinions
- Change the language (keep original language)
- Add headers

Return ONLY the cleaned transcript, no commentary.

Transcript:
${output}"""
