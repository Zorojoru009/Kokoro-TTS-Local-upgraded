# Kokoro TTS Prosody & Pronunciation Guide

This guide documents the specific syntax used by the Kokoro TTS engine (v0.19+) for controlling stress, intonation, and phonetic pronunciation. This information is crucial for crafting scripts that require precise delivery, such as dramatic readings, educational content, or character acting.

## 1. Stress & Intonation Control

Kokoro uses a Markdown-like link syntax to adjust the stress level of specific words or phrases. This mechanism influences the prosody (rhythm, stress, and intonation) of the generated speech.

### Syntax
`[text](level)`

*   **`text`**: The word or phrase to modify.
*   **`level`**: An integer value indicating the direction and magnitude of the stress change.

### Stress Levels

| Level | Effect | Usage Description | Example |
| :--- | :--- | :--- | :--- |
| **`(+2)`** | **Heavy Emphasis** | Used for shouting, authoritative commands, or the most critical word in a sentence. Often raises pitch and volume significantly. | `[STOP](+2) right there!` |
| **`(+1)`** | **Moderate Emphasis** | Used for key terms, active verbs, or highlighting a specific point. Adds natural inflection and weight. | `This is [important](+1).` |
| **`(0)`** | **Neutral** | Default behavior (no tag needed). | `The cat sat on the mat.` |
| **`(-1)`** | **Reduced Stress** | Used for throwaway words, quick asides, or to make something sound weak/insignificant. | `It was [just](-1) a mistake.` |
| **`(-2)`** | **Minimal Stress** | Used for very quiet, mumbled, or dismissive text. | `I [guess](-2) so.` |

### Best Practices
*   **Subtlety:** The effect of `(+1)` can sometimes be subtle. If a word needs to stand out clearly, `(+2)` is often more effective.
*   **Context:** Stress markers work best on content words (nouns, verbs, adjectives) rather than function words (the, a, is), unless you are specifically correcting someone (e.g., "I said *the* car, not *a* car").

---

## 2. Phonetic Override (IPA)

You can force a specific pronunciation for a word by providing its International Phonetic Alphabet (IPA) representation. This is essential for proper names, technical jargon, or invented words.

### Syntax
`[text](/ipa/)`

*   **`text`**: The display text (this is what is conceptually being spoken, but the audio is generated from the IPA).
*   **`/ipa/`**: The IPA string, enclosed in forward slashes.

### Important Notes on IPA
*   **Supported Symbols:** Kokoro uses a standard set of IPA symbols. Ensure you are using valid IPA characters.
*   **Stress Markers:** You *must* include stress markers within the IPA string for multi-syllabic words to ensure correct rhythm.
    *   `ˈ` (U+02C8) : Primary Stress
    *   `ˌ` (U+02CC) : Secondary Stress

### Examples

| Word | IPA Syntax | Description |
| :--- | :--- | :--- |
| **Kokoro** | `[Kokoro](/kˈOkəɹO/)` | Custom pronunciation for the engine name. |
| **Teleology** | `[Teleology](/ˌtɛliˈɒlədʒi/)` | Correct pronunciation of a complex philosophical term. |
| **Adler** | `[Adler](/ˈædlər/)` | Ensuring the German/English name is pronounced correctly. |
| **Resume** | `[resume](/rɪˈzuːm/)` | Distinguishing "resume" (verb) from "résumé" (noun). |

---

## 3. Punctuation & Timing

While Kokoro does not support SSML tags like `<break time="1s"/>`, punctuation is the primary tool for controlling pacing.

| Symbol | Effect | Usage |
| :--- | :--- | :--- |
| **`.`** | **Full Stop** | Standard pause. Lowers pitch at the end of the sentence. |
| **`,`** | **Comma** | Short pause. Slight upward inflection (continuation). |
| **`...`** | **Ellipsis** | Longer, dramatic pause. Often implies trailing off or suspense. |
| **`?`** | **Question Mark** | Raises pitch at the end for a question. |
| **`!`** | **Exclamation** | Increases volume and energy. |
| **`—`** | **Em Dash** | Sharp break or interruption. |

---

## 4. Combined Usage Example

You can mix these techniques to create highly expressive scripts.

**Script:**
"You think this is a [game](-1)? [No](+2). This is [Science](/ˈsaɪəns/)... and it is [dangerous](+1)."

**Breakdown:**
1.  **`[game](-1)`**: The word "game" is de-emphasized to show disdain.
2.  **`[No](+2)`**: The refusal is shouted or spoken with force.
3.  **`[Science](/ˈsaɪəns/)`**: Ensures "Science" is pronounced clearly (though standard English usually doesn't need IPA, this is for demonstration).
4.  **`...`**: A dramatic pause before the final warning.
5.  **`[dangerous](+1)`**: The final adjective is stressed to land the warning.
