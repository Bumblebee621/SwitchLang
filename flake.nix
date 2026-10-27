{
  description = "SwitchLang English/Hebrew keyboard layout auto-switcher";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        rec {
          switchlang = pkgs.callPackage ./nix/package.nix { };
          default = switchlang;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.switchlang}/bin/switchlang";
          meta.description = "English/Hebrew keyboard layout auto-switcher";
        };
      });

      checks = forAllSystems (system: {
        inherit (self.packages.${system}) switchlang;
      });

      nixosModules.default = import ./nix/module.nix;
      nixosModules.switchlang = self.nixosModules.default;
    };
}
