// Build-time generated distributed-slice entries.  An empty file is valid on
// CPU-only machines; runtime JIT behavior remains unchanged unless the AOT
// requirement flag is enabled in the local cuTile fork.
#[doc(hidden)]
mod aot_registry {
    include!(concat!(env!("OUT_DIR"), "/cutile_aot_registry.rs"));
}

/// Parse the comma-separated CUDA SM list used by the optional AOT build.
///
/// The default keeps the bridge usable on the repository's Ada development
/// GPU while allowing a release build to embed several architectures.  The
/// parser is intentionally shared with host-only tests; build-time compiler
/// availability is handled by `build.rs`.
pub fn parse_aot_target_sms(raw: Option<&str>) -> Result<Vec<String>, String> {
    let value = raw.unwrap_or("sm_89");
    let mut targets = Vec::new();
    for target in value.split(',').map(str::trim) {
        if target.is_empty() {
            return Err("LIGHTNET_CUTILE_AOT_SMS contains an empty target".to_owned());
        }
        let Some(digits) = target.strip_prefix("sm_") else {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        }
        if !targets.iter().any(|existing| existing == target) {
            targets.push(target.to_owned());
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod aot_registry_tests {
    use super::parse_aot_target_sms;

    #[test]
    fn parses_default_and_deduplicates_targets() {
        assert_eq!(parse_aot_target_sms(None).unwrap(), ["sm_89"]);
        assert_eq!(
            parse_aot_target_sms(Some(" sm_89,sm_90,sm_89 ")).unwrap(),
            ["sm_89", "sm_90"]
        );
    }

    #[test]
    fn rejects_malformed_targets() {
        assert!(parse_aot_target_sms(Some("sm_89,,sm_90")).is_err());
        assert!(parse_aot_target_sms(Some("89")).is_err());
        assert!(parse_aot_target_sms(Some("sm_")).is_err());
    }
}
