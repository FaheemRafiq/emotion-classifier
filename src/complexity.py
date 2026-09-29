"""Definitions of "complex" emotional text, used to slice the frozen GoEmotions test set.

These rules were written BEFORE any v3 model was evaluated on the test set, and are applied identically
to every model, so the slices cannot be tuned to favour one model. Each rule is deliberately simple and
transparent (a regular expression), so an examiner can check which texts fall into a slice.

  negation      - the sentence contains a negation ("not happy", "never again", "can't sleep")
  contrast      - a contrast word that often signals mixed or shifting feelings ("excited but scared")
  implicit      - no explicit emotion word at all: the emotion must be inferred from the situation
                  ("I skipped the gym again, third time this week")
  long          - 25 words or more (several clauses to combine)
  sarcasm_cue   - common sarcasm / irony markers ("/s", "yeah right", "oh great", "thanks a lot")
  emoji         - contains an emoji or emoticon
"""
import re

NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|cannot|without|hardly|barely)\b|n't\b", re.I)
CONTRAST = re.compile(r"\b(but|although|though|however|yet|despite|even though|whereas|still)\b", re.I)
SARCASM = re.compile(r"(/s\b|\byeah,? right\b|\boh,? (great|wonderful|perfect|joy)\b|\bjust (great|perfect|what i needed)\b"
                     r"|\bthanks a lot\b|\bwhat a (surprise|shock)\b|\bgood for you\b|\bsure,? (jan|buddy)\b"
                     r"|\blove (how|that|it when)\b.*\b(not|never|again|always)\b)", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|[:;]-?[()DPp]|<3")

# Explicit emotion vocabulary (stems). A text with none of these must be understood from context.
EMOTION_WORDS = re.compile(
    r"\b(happ\w*|glad|joy\w*|love\w*|lovin\w*|excit\w*|amaz\w*|awesome|great|wonderful|fantastic|delight\w*|"
    r"thank\w*|grateful|gratitude|relie\w*|proud|pride|hope\w*|hoping|optimis\w*|motivat\w*|inspir\w*|"
    r"sad\w*|cry\w*|cried|tears?|depress\w*|upset|hurt\w*|heartbr\w*|miserable|lonely|grie\w*|miss(ing)?|"
    r"scar\w*|afraid|fear\w*|terrif\w*|anxi\w*|nervous\w*|worr\w*|panic\w*|dread\w*|"
    r"ang(er|ry)\w*|mad|furious|rage|annoy\w*|irritat\w*|hate\w*|frustrat\w*|disgust\w*|gross|disappoint\w*|"
    r"guilt\w*|sorry|ashamed|shame\w*|embarrass\w*|regret\w*|remorse\w*|awkward|"
    r"lol|lmao|lmfao|rofl|xd|haha\w*|hehe\w*|wow|yay|ugh|omg|wtf|fun\w*|enjoy\w*|pleas(ed|ure)|bored|tired|stress\w*|"
    r"nice|cool|beautiful|cute|sweet|best|worst|awful|terrible|horrible|congrat\w*|luck\w*|damn\w*|fuck\w*|shit\w*)\b",
    re.I,
)


def slices(text: str) -> dict[str, bool]:
    words = text.split()
    return {
        "negation": bool(NEGATION.search(text)),
        "contrast": bool(CONTRAST.search(text)),
        "implicit": not EMOTION_WORDS.search(text) and not EMOJI.search(text),
        "long": len(words) >= 25,
        "sarcasm_cue": bool(SARCASM.search(text)),
        "emoji": bool(EMOJI.search(text)),
    }


SLICE_NAMES = ["negation", "contrast", "implicit", "long", "sarcasm_cue", "emoji"]
