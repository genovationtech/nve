//! NVE startup banner and model header.

/// Print the full NVE banner with box and ASCII logo.
pub fn print_banner() {
    let banner = "\
┌─────────────────────────────────────────────────────────────────┐\n\
│                                                                 │\n\
│   ███╗   ██╗██╗   ██╗███████╗                                  │\n\
│   ████╗  ██║██║   ██║██╔════╝                                  │\n\
│   ██╔██╗ ██║██║   ██║█████╗                                    │\n\
│   ██║╚██╗██║╚██╗ ██╔╝██╔══╝                                    │\n\
│   ██║ ╚████║ ╚████╔╝ ███████╗                                  │\n\
│   ╚═╝  ╚═══╝  ╚═══╝  ╚══════╝                                  │\n\
│                                                                 │\n\
│   Neural Virtualization Engine  v0.1.0                         │\n\
│   Genovation Technological Solutions Pvt Ltd                    │\n\
└─────────────────────────────────────────────────────────────────┘";
    println!("{}", banner);
}

/// Print a compact single-line header bar shown inside the chat REPL.
pub fn print_model_header(model_name: &str, quant_mode: &str, tier_mode: &str) {
    println!(
        "  ─── Model: {}  │  Quant: {}  │  Mode: {} ───",
        model_name, quant_mode, tier_mode
    );
}
