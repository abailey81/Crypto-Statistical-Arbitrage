# Security Policy

## API Key Handling

This project interacts with 32 cryptocurrency venues and data providers. Proper handling of API credentials is critical.

### Best Practices

- **Never commit API keys** to version control. The `.gitignore` already excludes `.env` files and common credential patterns.
- **Use the template**: Copy `config/api_keys_template.env` to `config/.env` and fill in your keys there.
- **Restrict API permissions**: When creating API keys on exchanges, use read-only permissions where possible. This project only reads market data -- it does not execute live trades.
- **Rotate keys regularly**: If you suspect a key has been exposed, revoke it immediately on the exchange and generate a new one.
- **Use IP whitelisting**: Most exchanges support restricting API keys to specific IP addresses. Enable this when possible.

### What the Project Does NOT Do

- This project does **not** execute live trades
- This project does **not** transfer funds
- This project does **not** store credentials in any database or cache
- API keys are only loaded from environment variables at runtime

---

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email the maintainer directly at the address listed in the GitHub profile: [abailey81](https://github.com/abailey81).
3. Alternatively, use [GitHub Security Advisories](https://github.com/abailey81/Crypto-Statistical-Arbitrage/security/advisories/new) to report privately.

### What to Include

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (optional but appreciated)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix**: Dependent on severity, but critical issues will be prioritized

---

## Supported Versions

| Version | Supported |
|:--------|:----------|
| Latest (`main`) | Yes |
| Older commits | Best effort |

---

## Scope

The following are in scope for security reports:

- Credential exposure in code, logs, or output files
- Insecure default configurations
- Dependency vulnerabilities with known exploits
- Data exfiltration risks

The following are **out of scope**:

- Vulnerabilities in third-party exchange APIs
- Issues requiring physical access to the host machine
- Social engineering attacks
