package com.convaiinnovations.laya.tokenizer;

import com.convaiinnovations.laya.json.Json;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The pre-tokenizer stage: it cuts normalized text into the pieces BPE then merges independently.
 *
 * <p>The two laya checkpoints take the two different routes a BPE model can take, and the
 * difference is not cosmetic -- it decides what a "word boundary" even is:
 *
 * <ul>
 *   <li><b>english</b>: {@code {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets":
 *       true, "use_regex": true}}. Text is cut by the GPT-2 regex and then each piece's UTF-8
 *       <i>bytes</i> are mapped into a 256-character printable alphabet, so a space becomes
 *       {@code U+0120} ({@code 'Ġ'}).
 *   <li><b>multilingual</b>: {@code {"type": "Metaspace", "replacement": "▁",
 *       "prepend_scheme": "always", "split": true}}. Spaces become {@code U+2581} and the text is
 *       cut before each one, so the marker rides on the front of the following word.
 * </ul>
 *
 * <p>Both are implemented here rather than one of them, because a port that assumed GPT-2 defaults
 * -- the obvious assumption, and the wrong one -- would mis-tokenize every multilingual input.
 */
abstract class PreTokenizer {

    /** Cuts one span of normalized text into pieces for the BPE stage. */
    abstract List<String> preTokenize(String text);

    /** Builds a pre-tokenizer from a checkpoint's {@code pre_tokenizer} node, which may be null. */
    static PreTokenizer from(Map<String, Object> node) {
        if (node == null) {
            return new PreTokenizer() {
                @Override
                List<String> preTokenize(String text) {
                    return text.isEmpty() ? List.of() : List.of(text);
                }
            };
        }
        String type = Json.string(node, "type", null);
        if (type == null) {
            throw new Json.JsonException("a pre-tokenizer must declare a type");
        }
        switch (type) {
            case "ByteLevel":
                return new ByteLevel(
                        Json.bool(node, "add_prefix_space", true),
                        Json.bool(node, "use_regex", true));
            case "Metaspace":
                return new Metaspace(
                        Json.string(node, "replacement", "▁"),
                        Json.string(node, "prepend_scheme", "always"),
                        Json.bool(node, "split", true));
            case "Sequence":
                List<PreTokenizer> steps = new ArrayList<>();
                for (Object child : Json.array(node, "pretokenizers")) {
                    if (!(child instanceof Map)) {
                        throw new Json.JsonException("a pre-tokenizer in a Sequence must be an object");
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> asMap = (Map<String, Object>) child;
                    steps.add(from(asMap));
                }
                return new PreTokenizer() {
                    @Override
                    List<String> preTokenize(String text) {
                        List<String> current = List.of(text);
                        for (PreTokenizer step : steps) {
                            List<String> next = new ArrayList<>(current.size());
                            for (String piece : current) {
                                next.addAll(step.preTokenize(piece));
                            }
                            current = next;
                        }
                        return current;
                    }
                };
            default:
                throw new Json.JsonException(
                        "this checkpoint uses the pre-tokenizer \"" + type + "\", which laya-java "
                        + "does not implement. Refusing to load rather than produce token ids that "
                        + "are wrong in a way no answer-level test would localise.");
        }
    }

    // ------------------------------------------------------------------ ByteLevel

    /** GPT-2 byte-level pre-tokenization: regex cut, then UTF-8 bytes into a printable alphabet. */
    static final class ByteLevel extends PreTokenizer {

        /**
         * The GPT-2 cut. Ordered alternation, and the order is the specification: {@code \s+(?!\S)}
         * must be tried before {@code \s+} so that a run of whitespace followed by a non-space
         * leaves its last space to the following word, which is what produces {@code "Ġthere"}
         * rather than a lone whitespace token.
         *
         * <p>{@link Pattern#UNICODE_CHARACTER_CLASS} is not optional: Java's {@code \s} is
         * ASCII-only by default, where the Rust implementation's is the Unicode
         * {@code White_Space} property. Without the flag, a non-breaking space or an ideographic
         * space would fall through to the punctuation branch and tokenize differently.
         */
        private static final Pattern GPT2 = Pattern.compile(
                "'s|'t|'re|'ve|'m|'ll|'d"
                + "| ?\\p{L}+"
                + "| ?\\p{N}+"
                + "| ?[^\\s\\p{L}\\p{N}]+"
                + "|\\s+(?!\\S)"
                + "|\\s+",
                Pattern.UNICODE_CHARACTER_CLASS);

        /** byte value -> the character that stands for it. */
        private static final char[] BYTE_TO_CHAR = buildByteToChar();

        private final boolean addPrefixSpace;
        private final boolean useRegex;

        ByteLevel(boolean addPrefixSpace, boolean useRegex) {
            this.addPrefixSpace = addPrefixSpace;
            this.useRegex = useRegex;
        }

        /**
         * The GPT-2 alphabet, built rather than transcribed.
         *
         * <p>188 byte values that are already printable stand for themselves; the remaining 68 are
         * lifted into {@code U+0100} and upward in ascending byte order. Transcribing the resulting
         * 256-character table into a literal is the kind of thing that is wrong in one position and
         * passes every test that does not happen to contain that byte, so it is derived here by the
         * same construction the reference implementation uses.
         */
        private static char[] buildByteToChar() {
            boolean[] printable = new boolean[256];
            for (int b = '!'; b <= '~'; b++) {
                printable[b] = true;
            }
            for (int b = 0xA1; b <= 0xAC; b++) {
                printable[b] = true;
            }
            for (int b = 0xAE; b <= 0xFF; b++) {
                printable[b] = true;
            }
            char[] table = new char[256];
            int lifted = 0;
            for (int b = 0; b < 256; b++) {
                if (printable[b]) {
                    table[b] = (char) b;
                } else {
                    table[b] = (char) (256 + lifted);
                    lifted++;
                }
            }
            if (lifted != 68) {
                throw new IllegalStateException(
                        "the byte-level alphabet should lift exactly 68 bytes, lifted " + lifted);
            }
            return table;
        }

        /**
         * The characters this pre-tokenizer can actually emit, for the load-time alphabet check.
         *
         * <p>243, not 256. Thirteen byte values -- {@code 0xC0}, {@code 0xC1} and {@code 0xF5}
         * through {@code 0xFF} -- cannot occur in the UTF-8 encoding of any text: the first two are
         * overlong two-byte leads and the rest encode nothing at or below {@code U+10FFFF}. Their
         * stand-in characters therefore never reach the vocabulary lookup.
         *
         * <p>This is not a theoretical nicety. The english checkpoint is missing <b>exactly those
         * thirteen</b> symbols from its 50,280-entry vocabulary and declares no unknown token, so a
         * check over all 256 rejects the shipped checkpoint outright. Verified by exhaustion over
         * every valid code point ({@code U+0000}-{@code U+10FFFF} less the surrogate range): 243
         * distinct bytes are reachable, and the thirteen absent symbols are precisely the thirteen
         * unreachable ones.
         */
        static List<String> alphabetReachableFromText() {
            List<String> out = new ArrayList<>(243);
            for (int b = 0; b < 256; b++) {
                if (b == 0xC0 || b == 0xC1 || b >= 0xF5) {
                    continue;
                }
                out.add(String.valueOf(BYTE_TO_CHAR[b]));
            }
            return out;
        }

        @Override
        List<String> preTokenize(String text) {
            String source = text;
            if (addPrefixSpace && !source.startsWith(" ")) {
                source = " " + source;
            }
            List<String> out = new ArrayList<>();
            if (!useRegex) {
                if (!source.isEmpty()) {
                    out.add(encode(source));
                }
                return out;
            }
            Matcher matcher = GPT2.matcher(source);
            while (matcher.find()) {
                if (matcher.end() > matcher.start()) {
                    out.add(encode(source.substring(matcher.start(), matcher.end())));
                }
            }
            return out;
        }

        /** One piece's UTF-8 bytes, each replaced by its stand-in character. */
        private static String encode(String piece) {
            byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
            StringBuilder out = new StringBuilder(bytes.length);
            for (byte b : bytes) {
                out.append(BYTE_TO_CHAR[b & 0xFF]);
            }
            return out.toString();
        }
    }

    // ------------------------------------------------------------------ Metaspace

    /** SentencePiece-style pre-tokenization: a space becomes a visible marker on the next word. */
    static final class Metaspace extends PreTokenizer {

        private final String replacement;
        private final boolean prependAlways;
        private final boolean prependFirst;
        private final boolean split;

        Metaspace(String replacement, String prependScheme, boolean split) {
            if (replacement == null || replacement.codePointCount(0, replacement.length()) != 1) {
                throw new Json.JsonException(
                        "a Metaspace replacement must be exactly one code point, got "
                        + (replacement == null ? "null" : "\"" + replacement + "\""));
            }
            this.replacement = replacement;
            switch (prependScheme) {
                case "always":
                    this.prependAlways = true;
                    this.prependFirst = false;
                    break;
                case "first":
                    this.prependAlways = false;
                    this.prependFirst = true;
                    break;
                case "never":
                    this.prependAlways = false;
                    this.prependFirst = false;
                    break;
                default:
                    throw new Json.JsonException(
                            "unknown Metaspace prepend_scheme \"" + prependScheme + "\"");
            }
            this.split = split;
        }

        @Override
        List<String> preTokenize(String text) {
            if (text.isEmpty()) {
                return List.of();
            }
            String source = text.replace(" ", replacement);
            // Prepend only when the text does not ALREADY start with the marker. The guard is not
            // an optimisation; without it the multilingual checkpoint is wrong on any span that
            // begins with a space, because its normalizer has already turned that space into the
            // marker and prepending a second one invents a token. Measured against the reference
            // implementation: " " encodes to ONE marker token, not two, and "   leading spaces"
            // to four ids, not five.
            //
            // `first` differs from `always` only across a multi-span sequence, and laya feeds this
            // one span at a time, so the two coincide here. Kept distinct so a checkpoint
            // declaring `first` is not silently given `always` semantics if a caller ever batches.
            if ((prependAlways || prependFirst) && !source.startsWith(replacement)) {
                source = replacement + source;
            }
            if (!split) {
                return List.of(source);
            }
            // Cut *before* each marker, so the marker leads the piece it belongs to. A leading
            // marker must not produce an empty first piece.
            List<String> out = new ArrayList<>();
            int start = 0;
            int at = source.indexOf(replacement, 1);
            while (at >= 0) {
                out.add(source.substring(start, at));
                start = at;
                at = source.indexOf(replacement, at + replacement.length());
            }
            if (start < source.length()) {
                out.add(source.substring(start));
            }
            return out;
        }
    }
}
