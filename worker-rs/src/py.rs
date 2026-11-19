//! Python-compatible helpers: `json.dumps` (separators, `indent`,
//! `ensure_ascii`), `repr(float)`, `shlex.join`, `urllib.parse.quote`,
//! `strftime` in UTC, `shutil.which`, `round(x, n)`. The worker's output
//! strings (the `command` line, `job.json`, the local-mode result) are
//! byte-compatible with the Python handler's through these.

use serde_json::{Map, Number, Value};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// `repr(float)` for a finite f64: the shortest round-trip digits, fixed
/// notation for exponents in [-4, 16), else `1e+16` / `1e-05`.
pub fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // `{:e}` gives the shortest round-trip mantissa: "1.5e2", "-3e-7".
    let s = format!("{:e}", v);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..(-point) {
                out.push('0');
            }
            out.push_str(&digits);
        } else if (point as usize) >= digits.len() {
            out.push_str(&digits);
            for _ in 0..(point as usize - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// `f"{v:.6g}"` (the flag builder's float format).
pub fn format_g6(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if !v.is_finite() {
        return float_repr(v);
    }
    let s = format!("{:.5e}", v); // 6 significant digits, rounded
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= 6 {
        let m = mant.trim_end_matches('0').trim_end_matches('.');
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        let f = format!("{:.*}", decimals, v);
        if f.contains('.') {
            f.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            f
        }
    }
}

/// `str(number)` / the text a JSON number has in Python.
pub fn number_str(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        float_repr(n.as_f64().unwrap_or(0.0))
    }
}

/// `str(value)` for a scalar.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => number_str(n),
        Value::String(s) => s.clone(),
        other => dumps(other, None, true),
    }
}

/// `repr(value)` for a scalar (`'vsa'`, `None`, `0.8`).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => py_str_repr(s),
        other => py_str(other),
    }
}

/// `repr(str)`: single quotes unless the string holds one and no double quote.
pub fn py_str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `a == b` with Python's number semantics (`1 == 1.0`).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Bool(x), Value::Number(y)) | (Value::Number(y), Value::Bool(x)) => {
            y.as_f64() == Some(if *x { 1.0 } else { 0.0 })
        }
        _ => a == b,
    }
}

/// `isinstance(v, int) and not isinstance(v, bool)`.
pub fn is_int(v: &Value) -> bool {
    matches!(v, Value::Number(n) if n.is_i64() || n.is_u64())
}

/// `round(x, ndigits)`.
pub fn round_to(x: f64, ndigits: i32) -> f64 {
    let p = 10f64.powi(ndigits);
    let r = (x * p).round() / p;
    if r.is_finite() {
        r
    } else {
        x
    }
}

fn write_json_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", u);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(value, indent=indent, ensure_ascii=ensure_ascii)`: Python's
/// separators (`, ` / `: ` compact, `,` + newline / `: ` indented),
/// insertion order, `repr` floats.
pub fn dumps(value: &Value, indent: Option<usize>, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, value, indent, ensure_ascii, 0);
    out
}

fn write_value(
    out: &mut String,
    v: &Value,
    indent: Option<usize>,
    ensure_ascii: bool,
    level: usize,
) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| !n.is_i64() && !n.is_u64()) {
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&float_repr(f));
                }
            } else {
                out.push_str(&number_str(n));
            }
        }
        Value::String(s) => write_json_str(out, s, ensure_ascii),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_value(out, x, indent, ensure_ascii, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Object(o) => {
            if o.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_json_str(out, k, ensure_ascii);
                out.push_str(": ");
                write_value(out, x, indent, ensure_ascii, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        for _ in 0..(n * level) {
            out.push(' ');
        }
    }
}

/// `shlex.quote`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

/// `shlex.join`.
pub fn shlex_join<S: AsRef<str>>(argv: &[S]) -> String {
    argv.iter()
        .map(|a| shlex_quote(a.as_ref()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `urllib.parse.quote(s, safe=safe)`: UTF-8 bytes, `-_.~` and ASCII
/// alphanumerics always kept.
pub fn quote(s: &str, safe: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        let c = b as char;
        if b.is_ascii_alphanumeric() || "-_.~".contains(c) || (b < 0x80 && safe.contains(c)) {
            out.push(c);
        } else {
            let _ = write!(out, "%{:02X}", b);
        }
    }
    out
}

/// `(year, month, day, hour, minute, second)` of a Unix time in UTC
/// (Howard Hinnant's civil-from-days).
pub fn utc_parts(unix_s: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = unix_s.div_euclid(86_400);
    let rem = unix_s.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        m,
        d,
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// `strftime("%Y%m%dT%H%M%SZ")` (SigV4's `X-Amz-Date`).
pub fn amz_date(unix_s: i64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(unix_s);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

/// `strftime("%Y-%m-%dT%H:%M:%SZ")` (`expires_at`).
pub fn iso_utc(unix_s: i64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(unix_s);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `shutil.which`.
pub fn which(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let p = Path::new(name);
        return if is_executable(p) {
            Some(p.to_path_buf())
        } else {
            None
        };
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if is_executable(&p) {
            return Some(p);
        }
    }
    None
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

/// `os.path.realpath`: resolves what exists and keeps the rest of the path.
pub fn realpath(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    let mut rest = Vec::new();
    let mut cur = p.to_path_buf();
    loop {
        if let Ok(c) = std::fs::canonicalize(&cur) {
            let mut out = c;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (
            cur.file_name().map(|f| f.to_os_string()),
            cur.parent().map(|q| q.to_path_buf()),
        ) {
            (Some(name), Some(parent)) if !parent.as_os_str().is_empty() => {
                rest.push(name);
                cur = parent;
            }
            _ => {
                let mut out = if p.is_absolute() {
                    PathBuf::new()
                } else {
                    std::env::current_dir().unwrap_or_default()
                };
                out.push(p);
                return out;
            }
        }
    }
}

/// A JSON object with Python's insertion order.
pub fn obj() -> Map<String, Value> {
    Map::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (v, want) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (12.1, "12.1"),
            (100.0, "100.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5, "-2.5"),
            (1.5e-7, "1.5e-07"),
            (123456789.125, "123456789.125"),
            (0.1 + 0.2, "0.30000000000000004"),
        ] {
            assert_eq!(float_repr(v), want, "{v}");
        }
    }

    #[test]
    fn g6_matches_python() {
        for (v, want) in [
            (0.9, "0.9"),
            (1.0, "1"),
            (0.85, "0.85"),
            (1234567.0, "1.23457e+06"),
            (0.00001234, "1.234e-05"),
            (2.5, "2.5"),
            (100.0, "100"),
        ] {
            assert_eq!(format_g6(v), want, "{v}");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v =
            json!({"a": 1, "b": [1.5, "x", null, true], "c": {}, "d": [], "e": "ü✓", "f": "q\"\n"});
        assert_eq!(
            dumps(&v, None, false),
            r#"{"a": 1, "b": [1.5, "x", null, true], "c": {}, "d": [], "e": "ü✓", "f": "q\"\n"}"#
        );
        assert_eq!(
            dumps(&json!({"e": "ü✓𝄞"}), None, true),
            "{\"e\": \"\\u00fc\\u2713\\ud834\\udd1e\"}"
        );
        assert_eq!(
            dumps(&json!({"a": [1, {"b": 2}], "c": {}}), Some(1), true),
            "{\n \"a\": [\n  1,\n  {\n   \"b\": 2\n  }\n ],\n \"c\": {}\n}"
        );
    }

    #[test]
    fn shlex_and_quote() {
        assert_eq!(
            shlex_join(&["h3", "generate", "--prompt", "it's a 'fox'"]),
            "h3 generate --prompt 'it'\"'\"'s a '\"'\"'fox'\"'\"''"
        );
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(
            quote("uploads/a b/ü (1)+x&y=z#!'*.png", "/"),
            "uploads/a%20b/%C3%BC%20%281%29%2Bx%26y%3Dz%23%21%27%2A.png"
        );
        assert_eq!(quote("a/b", ""), "a%2Fb");
        assert_eq!(
            quote("fvstudio-worker/0.1.0", ""),
            "fvstudio-worker%2F0.1.0"
        );
    }

    #[test]
    fn utc_formats() {
        // 2026-11-19T12:34:56Z
        assert_eq!(amz_date(1_795_091_696), "20261119T123456Z");
        assert_eq!(iso_utc(1_795_091_696), "2026-11-19T12:34:56Z");
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(amz_date(1_440_938_160), "20150830T123600Z");
    }

    #[test]
    fn repr_and_misc() {
        assert_eq!(py_repr(&json!("vsa")), "'vsa'");
        assert_eq!(py_repr(&json!(null)), "None");
        assert_eq!(py_repr(&json!(0.8)), "0.8");
        assert_eq!(py_repr(&json!("it's")), "\"it's\"");
        assert!(py_eq(&json!(1), &json!(1.0)) && !py_eq(&json!("1"), &json!(1)));
        assert_eq!(round_to(2.0416666, 3), 2.042);
        assert!(is_int(&json!(5)) && !is_int(&json!(5.0)) && !is_int(&json!(true)));
    }
}
