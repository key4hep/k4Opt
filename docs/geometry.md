# Geometry component

`k4opt.geometry` discovers and overrides parameters in DD4hep / Key4hep XML
detector descriptions and materializes modified copies of a geometry. It is the
first component of k4opt; later components (problem definition with bounds,
evaluation, optimization) build on its public API (`from k4opt.geometry import ...`).

What it does:

1. names a geometry the way Key4hep does, e.g. `FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml`,
2. lists every XML attribute reachable from that file as a readable, copy-pastable *target reference*,
3. overrides selected attributes with literal replacement strings,
4. materializes an isolated, modified copy of the geometry together with a JSON manifest,

without touching the installed geometry, without annotating the XML, and without
writing XPath. What it deliberately does not do: evaluate DD4hep expressions,
decide which parameters matter or how far they may vary, run simulations or
validate that the result is a sensible detector.

## Usage

A configuration file names the geometry and, optionally, the overrides
(see [`examples/geometry/cld.yaml`](../examples/geometry/cld.yaml)):

```yaml
geometry: FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml   # relative to $K4GEO, like k4Bench

overrides:                                              # optional for `inspect`
  - target: constants.InnerTracker_Barrel_radius_1
    value: "405*mm"
  - target: InnerTrackerBarrel.layer[id=1].rphi_layout.nphi
    value: "90"
```

```bash
# list addressable parameters (target, raw value, flags, file)
k4opt geometry inspect examples/geometry/cld.yaml
k4opt geometry inspect examples/geometry/cld.yaml --filter InnerTracker --format json

# preview the resolved changes; nothing is written
k4opt geometry apply examples/geometry/cld.yaml --output ./generated/cld-test --dry-run

# create the modified snapshot
k4opt geometry apply examples/geometry/cld.yaml --output ./generated/cld-test
ddsim --compactFile ./generated/cld-test/FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml ...
```

`inspect` groups parameters by file, with a header (geometry, root, how many are
shown), a flag column (`A` ambiguous across files, `O` order-sensitive), the
target you can copy and the raw value:

```text
geometry  FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml
root      /cvmfs/.../share/k4geo
showing   3 of 6160 parameters in 22 files, matching 'tracking_volume.shape.zplane[1]'

Flags in the first column:
  O  order-sensitive: positional [index] in the target, breaks if siblings are reordered

FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml  (3)
  O  tracking_volume.shape.zplane[1].z     -OuterTracker_half_length
  O  tracking_volume.shape.zplane[1].rmin  149*mrad * ConeBeamPipe_zmax
  O  tracking_volume.shape.zplane[1].rmax  OuterTracker_outer_radius
```

A short legend (in yellow on a terminal) explains the flags that occur in the
listing and is omitted when no row is flagged.

On a terminal, long values are shortened with `…` to fit the width (targets
never are) and output is colored; when piped, nothing is shortened or colored.
`NO_COLOR=1` turns color off. Use `--format json` for machine-readable output;
it carries a `schema` field (`k4opt.geometry.inspect/1`, `k4opt.geometry.apply/1`).

`apply` lists each change as `old  ->  new` grouped by file, then prints the
snapshot root, the generated compact file and the manifest path. Exit codes: `0`
success, `1` any expected k4opt error (message on stderr, prefixed
`k4opt: error:`), `2` command-line usage error.

The example values are technical examples only; they are not validated detector
variants, and a well-formed XML document is not a valid detector.

### Where the geometry comes from

`geometry` is resolved like k4Bench's `xml:` key:

| `geometry` | root used for includes and `file:` paths |
| --- | --- |
| `FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml` (relative) | `$K4GEO` from the Key4hep stack |
| `$K4GEO/FCCee/...` or another `$VAR/...` | variables are expanded first, then as below |
| `/abs/path/k4geo/FCCee/.../CLD_o2_v09.xml` | `$K4GEO` if the file lies inside it, else the enclosing git checkout, else the file's directory |

To work on your own k4geo checkout add `root: ../k4geo` (relative to the YAML
file); `geometry` is then relative to `root`. Includes may use `../` as long as
they stay inside the root.

### Python API

```python
from k4opt.geometry import GeometrySource, Override

source = GeometrySource("FCCee/CLD/compact/CLD_o2_v09/CLD_o2_v09.xml")  # or root="/path/to/k4geo"

catalog = source.discover()  # read-only, indexed sequence of Parameter records
for p in catalog.filter("InnerTracker"):
    print(p.target, p.file, p.value, p.order_sensitive, p.ambiguous)

p = catalog.lookup(
    "InnerTrackerBarrel.layer[id=1].rphi_layout.nphi"
)  # -> Parameter(value="43*2", ...)

overrides = [
    Override(target="constants.InnerTracker_Barrel_radius_1", value="405*mm"),
    Override(target="InnerTrackerBarrel.layer[id=1].rphi_layout.nphi", value="90"),
]
for change in source.resolve(overrides, catalog):  # dry run
    print(change.file, change.target, change.old_value, "->", change.new_value)

result = source.materialize(overrides, output="./generated/cld-test", catalog=catalog)
print(result.compact_path, result.manifest_path, [f.path for f in result.files])
```

Errors derive from `k4opt.geometry.GeometryError` (itself a `k4opt.K4OptError`):
`SourceError`, `IncludeError`, `TargetError` (with `InvalidTargetError`, `UnknownTargetError`,
`AmbiguousTargetError`, `DuplicateTargetError`) and `MaterializationError`; a
malformed YAML file raises `k4opt.ConfigError`. When
several overrides fail at once a plain `TargetError` is raised whose `errors`
attribute lists each failure.

## Target references

A target names exactly one XML attribute. References are **generated by
discovery**; copy them from `inspect` instead of writing them by hand.

```text
reference = "constants." NAME                      # value attribute of <constant name="NAME">
          | anchor { "." step } "." ATTR
anchor    = NAME                                   # the element carrying name="NAME"
          | TAG "[name=" VALUE "]"                 # tag-qualified named element
          | TAG                                    # the document root element (e.g. lccdd)
step      = TAG                                    # only child with that tag
          | TAG "[name=" VALUE "]" | TAG "[id=" VALUE "]"
          | TAG "[" INDEX "]"                      # 1-based position among same-tag siblings
NAME, TAG, ATTR = [A-Za-z_][A-Za-z0-9_-]*      VALUE = [A-Za-z0-9_.+-]+
```

Examples from CLD (`FCCee/CLD/compact/CLD_o2_v09`):

| target | addresses |
| --- | --- |
| `constants.InnerTracker_Barrel_radius_1` | `value` of `<constant name="InnerTracker_Barrel_radius_1">` |
| `InnerTrackerBarrel.layer[id=1].rphi_layout.nphi` | `nphi` on `<rphi_layout>` inside `<layer id="1">` of `<detector name="InnerTrackerBarrel">` |
| `GlobalSolenoid.inner_field` | `inner_field` of `<field name="GlobalSolenoid">` |
| `tracking_volume.shape.zplane[2].rmax` | second `<zplane>` (order-sensitive) |
| `detector[name=CompSol].type` | the detector named `CompSol`, because another element in the same file also carries that name |
| `lccdd.include[3].ref` | third top-level `<include>` of a file (positional) |

How a reference is chosen, per attribute: `constants.NAME` for constant values;
otherwise the nearest ancestor-or-self with a usable `name` becomes the anchor
and the steps below it use `[name=…]`/`[id=…]` when unique among siblings, no
qualifier when the tag is unique, and `[index]` otherwise. If two attributes in
the same file would get the same text, both fall back to the next form:
tag-qualified anchor, then the next named ancestor further up, and finally a
positional path from the document root. The result is unique within each file by
construction.

Flags shown by `inspect`:

- **A** (`ambiguous`): the same reference exists in another reachable file. Add
  `file: <root-relative path>` to the override. In CLD, `constants.HCal_cell_size`
  is defined in both HCal files, and positional paths such as
  `lccdd.module_component[2].thickness` repeat across module files.
- **O** (`order_sensitive`): the reference contains a positional `[index]` and
  will change if sibling elements are inserted or reordered.

Names containing characters outside `[A-Za-z0-9_-]` (e.g. `name="Pixel array"`)
are never used as anchors or qualifiers; such elements get an index step, and a
constant with such a name has no `constants.` alias. Looking up a non-canonical
spelling (e.g. `X.value` for a constant, or an anchor that is ambiguous within a
file) fails with a message listing the canonical references to use.

## Include handling

Discovery starts at the geometry file and recursively parses local files
referenced by the `ref` attribute of `<include>`, `<gdmlFile>`, `<file>` and
`<xml>` elements, wherever they appear. Each physical file is parsed once;
cycles and repeated includes are safe. Relative references resolve against the
including file's directory. `ref` attributes on other elements that point at an
existing local file (e.g. a CAD `.stl` model) are recorded as *resources*.

References are **left intact and reported as external**, and their targets are
not part of the catalog, when they contain `$` (e.g.
`${DD4hepINSTALL}/DDDetectors/compact/detector_types.xml`), are URLs, are
absolute paths, or are empty. A *missing* literal local include, an include that
escapes the root (lexically or through a symlink), or one that is not
well-formed XML fails discovery with the include chain in the message.

Only the geometry file and its reachable files are scanned, not the whole tree.
This is an XML include walker, not an interpreter of DD4hep preprocessing or of
C++ detector constructors: files a detector plugin opens by itself, or includes
constructed at runtime, are not seen. XML is parsed with lxml in a hardened
configuration: no entity expansion, no DTD loading, no network access.

## Override semantics

- `value` is the literal replacement string for the attribute. Quote it in YAML
  (`"405*mm"`, `"2*tesla"`, `"90"`); numbers and booleans are rejected so the
  text is exactly what you wrote. Nothing is evaluated, converted or inferred.
- Changing a constant propagates wherever DD4hep evaluates that constant.
  Changing an attribute changes only that attribute. k4opt never rewrites
  dependent attributes.
- All overrides are resolved against the catalog of the *unmodified* source
  before anything is written: unknown, ambiguous and malformed targets, and two
  overrides selecting the same underlying attribute, are errors. All failures are
  reported together.
- Edits are structural (`lxml`): no regex or line-based replacement.

## Materialization and manifest

`apply` (or `GeometrySource.materialize`) performs these steps:

1. Checks that `--output` does not exist and does not lie inside the source tree
   (also through symlinks).
2. Copies the reachable files, and only those (the geometry file, every local
   XML it includes, and local resource files they reference), into a temporary
   sibling of the output directory, keeping their root-relative layout so
   relative includes keep working. Symlinks are followed; nothing else from the
   tree is copied, so the snapshot of a 60 MB k4geo installation is a few dozen
   XML files.
3. Re-parses each affected file from the copy, checks that the current value is
   the one discovery recorded, sets the new value, serializes, re-parses the
   result and verifies that nothing but the intended attributes changed.
4. Re-walks the include graph of the edited snapshot so every local include still
   resolves.
5. Writes `k4opt-manifest.json` and atomically renames the copy to `--output`.
   On any failure the temporary copy is deleted; source and destination are
   untouched.

Run the result with `ddsim --compactFile <output>/<geometry path>` while keeping
the stack's `$K4GEO` as it is: `${K4GEO}`/`${DD4hepINSTALL}` references inside
the XML continue to point at the unmodified installation, which is intended.

Unchanged files are byte-identical copies. Edited files are re-serialized by
lxml: comments, text, element order, attribute order and namespaces are kept,
but whitespace inside start tags and quote styles are normalized.

The manifest records the schema version, the k4opt version, the source root and
compact path, the git commit of the source tree plus a `dirty` flag when it is a
checkout (a commit alone does not identify a dirty tree), the reachable files,
resources and external includes, every change (target, file, attribute,
structural locator, line, old and new raw value) and the SHA-256 of each changed
file before and after. It also records the software stack (`KEY4HEP_STACK` and
`K4GEO`, when set), because unchanged files and `${K4GEO}` references come from
that installation.

### Fingerprint

Every materialization has a `fingerprint`: a SHA-256 over the compact path, the
content hashes of all reachable files and resources, and the *effective* edits
(overrides that actually change a value). It contains no timestamps or paths, so
equal fingerprints mean equal geometries, and an override that restates the
current value does not change it. Use it to deduplicate or cache evaluations:

```python
fingerprint = source.fingerprint(overrides, catalog)  # without writing anything
result = source.materialize(overrides, output, catalog)
assert result.fingerprint == fingerprint
```

### Parallel and batch use

`GeometrySource`, `Catalog`, `Override` and `MaterializationResult` can be
pickled, so they can be sent to worker processes or batch jobs. Discover once and
pass the `catalog` to each `materialize` call; a call with a catalog takes a few
tens of milliseconds for CLD. Every snapshot is a self-contained directory that
can be shipped to another machine, but `${K4GEO}` references inside it still
resolve on the machine that runs it.

## Known limitations

- Elements or attributes with XML namespaces or with names outside
  `[A-Za-z0-9_-]` are not addressable and are listed in `Catalog.unaddressable`.
- Element text content (e.g. `<id>system:5,...</id>`) is not addressable; only attributes are.
- Include forms other than `ref` attributes on `include`/`gdmlFile`/`file`/`xml`
  elements, `$`-expanded paths, URLs and absolute paths are not followed.
  Resources are only detected through `ref` attributes with a local relative path.
- No DD4hep expression evaluation, unit handling, overlap checking or geometry
  validation of any kind.
- Adding or removing XML elements, and editing C++ detector constructors, are out of scope.
