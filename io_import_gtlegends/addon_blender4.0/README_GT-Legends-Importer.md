# GT Legends Importer 0.8.0 (Autor: haunetal1990)

Importiert GT-Legends-Autos und -Strecken nach Blender (4.0+ als Add-on, 4.2+ als Extension).

**Menü:** File > Import > GT Legends (.GTL)
**Dateien:** `.CAR` (Fahrerdatei → wählt das passende `.CAS`), `.CAS` (Auto), `.TRK`/`.SCN` (Strecke), `.GTL`-Archive, `.GMT`.

## Import-Optionen
- **Import sky/skybox** (aus) – Himmelskuppel, Sonne, Mond.
- **Import lights** (an) – `Light=`-Blöcke (Nightlights) als Punktlichter in der eigenen Collection `<Name>_Nightlights`.
- **Use texture alpha** (an) – Texturen mit Alpha-Kanal → Alpha an Principled BSDF, Blend Mode *Alpha Clip*; sonst *Opaque*.

## Wichtige Regeln
- **Collections:** Strecken = eine Collection (+ `_Nightlights`). Autos = Unter-Collections `Max/High/Med/Low` (LODs) und `Spin/NotSpin/NotSpin_DashHigh/Expo`. High/Med/Low sind per **Auge** ausgeblendet („Hide in Viewport“); *Show in Viewports* bleibt aktiv.
- **`Render=False`** in der `.TRK` → Objekt wird per Auge ausgeblendet (und nicht gerendert).
- Jedes Mesh wird nur **einmal** importiert.
- **Materialien:** Principled BSDF, Roughness 1.0, Textur direkt in Base Color; fehlende Texturen = pinker Platzhalter.
- **Face Orientation:** Winding unverändert aus der GMT-Datei (durch die X-Spiegelung der Koordinaten-Umrechnung passt es automatisch).
- **Treiber (Autos):** Lenkrad-Pivot, ExternalDriver-/FirstPersonDriver-Position aus der `.CAR`.
- UVs (V gespiegelt), Vertex-Farben (`Col`), Tangent/Binormal-Attribute und Spiel-Normalen (Custom Split Normals) werden übernommen.

## Tempo
Archive werden nur indiziert und lazy dekomprimiert, GMTs direkt aus dem Speicher geparst, Mesh-Aufbau und Alpha-Erkennung per numpy.

## Einstellungen (Add-on Preferences)
Archive-Cache: Blender-Temp-Ordner (Standard), Ordner neben den Originaldateien oder eigener Ordner. Extrahiert werden nur Texturen.

## Installation
- Blender 4.0/4.1: `gt_legends_importer_blender4.0_addon.zip` über *Preferences > Add-ons > Install*.
- Blender 4.2+: `gt_legends_importer_blender4.2_extension.zip` über *Preferences > Get Extensions > Install from Disk*.
