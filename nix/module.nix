{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.switchlang;
in
{
  options.programs.switchlang = {
    enable = lib.mkEnableOption "SwitchLang keyboard layout correction";
    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { };
      description = "SwitchLang package to install";
    };
    autostart = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Start SwitchLang in users' graphical sessions";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package ];
    services.udev.packages = [ cfg.package ];
    boot.kernelModules = [ "uinput" ];

    # Keep the UI and manual launches aware of the declarative startup setting.
    environment.sessionVariables.SWITCHLANG_AUTOSTART_MANAGED = if cfg.autostart then "1" else "0";

    systemd.user.services.switchlang = lib.mkIf cfg.autostart {
      description = "SwitchLang keyboard layout auto-switcher";
      partOf = [ "graphical-session.target" ];
      after = [ "graphical-session.target" ];
      wantedBy = [ "graphical-session.target" ];
      environment.SWITCHLANG_AUTOSTART_MANAGED = "1";
      serviceConfig = {
        ExecStart = lib.getExe cfg.package;
        Restart = "on-failure";
        RestartSec = 3;
      };
    };
  };
}
