# Vendored Fabric JSON schemas

Microsoft's published JSON schemas for the PBIP project files, mirrored from
[`microsoft/json-schemas`](https://github.com/microsoft/json-schemas) at the URL path each one
declares as its `$id`. `tests/test_powerbi.py` validates every document
`src/powerbi/tmdl.py` generates against these.

**Why vendored rather than fetched.** Tests that reach the network fail on a plane and pass
against whatever the schema happens to say today. These are pinned, so a schema changing upstream
shows up as a deliberate re-vendor in a diff rather than as a test that quietly starts failing.

**Why here and not `powerbi/`.** That folder gets copied to a Windows machine wholesale to be
opened in Desktop. Test fixtures have no business travelling with the deliverable.

## Contents

The closure is nine documents — the seven the generator emits a `$schema` for, plus two more that
`report/1.0.0` and `page/1.0.0` reach through relative `$ref`s:

| Path | Validates |
|---|---|
| `fabric/pbip/pbipProperties/1.0.0` | `Northstar.pbip` |
| `fabric/item/report/definitionProperties/2.0.0` | `Northstar.Report/definition.pbir` |
| `fabric/gitIntegration/platformProperties/2.0.0` | both `.platform` files |
| `fabric/item/report/definition/versionMetadata/1.0.0` | `definition/version.json` |
| `fabric/item/report/definition/report/1.0.0` | `definition/report.json` |
| `fabric/item/report/definition/pagesMetadata/1.0.0` | `definition/pages/pages.json` |
| `fabric/item/report/definition/page/1.0.0` | the five `page.json` |
| `fabric/item/report/definition/semanticQuery/1.0.0` | transitive `$ref` only |
| `fabric/item/report/definition/formattingObjectDefinitions/1.0.0` | transitive `$ref` only |

## Re-vendoring

```bash
BASE=https://raw.githubusercontent.com/microsoft/json-schemas/main
curl -sf "$BASE/<path>/schema.json" -o tests/fixtures/fabric_schemas/<path>/schema.json
```

`test_every_generated_schema_url_is_vendored` fails if `src/powerbi/tmdl.py` starts emitting a
`$schema` with no file here — so bumping a schema version in the generator forces the mirror to
be updated rather than letting validation silently skip the document.
