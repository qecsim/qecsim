1.0
---

1.0b7
~~~~~

Tagged 2021-01-29

- docs: Add BSD 3-Clause license in readme and generated docs.
- docs: Update for PyPI installation in readme and generated docs.
- cfg: Update setup.cfg for license and finalize meta-data.

1.0b6
~~~~~

Tagged 2021-01-27

- src: Remove CLI restriction on max random-seed (unnecessary since Numpy 1.17).
- src: Extend ``app`` output with logical commutations and custom values.
- src: Improve decoder flexibility: decoders may partially or completely specify
  success, logical commutations and custom values via ``model.DecodeResult``.
- src: Update ``models.rotatedtoric.RotatedToricSMWPMDecoder`` to return
  ``model.DecodeResult`` with custom values for fault-tolerant decoding.
- src: Improve code/decoder separation: move all ``distance`` methods to
  decoders.
- docs: Add change log.

1.0b5
~~~~~

Tagged 2021-01-15

- src: Add module script to enable CLI with Python options such as optimize.
- src: Support non-IID error models: make ``probability_distribution`` method
  optional.
- src: Improve code/decoder separation: move ``distance`` methods to decoders.
- src: Improve decoder flexibility: add ``model.DecodeResult`` as optional
  ``decode`` return value.
- src: Improve SMWPM/CMWPM decoder extensibility: move functions to class
  methods.
- docs: Rename Acknowledgment section to Citing and update with DOI.
- docs: Add EQUS acknowledgement.
- docs: Add License and Citing sections in readme.
- docs: Update author with initials in docs and build.
- tox: Separate unit test and performance test tasks.

1.0b4
~~~~~

Tagged 2020-12-14

- src: Improve MPS decoder extensibility: separate H and V value methods.
- tox: Note docs task dependency on pandoc.

1.0b3
~~~~~

Tagged 2020-09-23

- src: Improve model class extensibility (ported from qecsim legacy 0.18beta3):

  - Fix type equality for subclasses.
  - Replace static methods with class methods.
  - Use dynamic type names in exceptions.
  - Move tensor-network creator classes inside MPS decoders.

- src: Reformat code according to lint style.
- tox: Define lint task.

1.0b2
~~~~~

Tagged 2020-09-21

- src: Validate tests on MacOS, Linux and Windows.
- docs: Document C++ fast-matching library installation.

1.0b1
~~~~~

Tagged 2020-07-04

- Port qecsim legacy 0.18beta2 to standard extensible packaging.
- build: Configure setuptools.
- src: Dynamically load models defined in ``setup.cfg[options.entry_points]``.
- src: Add ``model.cli_description`` for CLI model usage text.
- src: Separate core/models tests from application source code.
- src: Rename ``QecsimException`` to ``QecsimError``.
- src: Remove unused ``util.file_cache`` and tests for Windows compatibility.
- docs: Add Acknowledgments, License, Links sections.
- tox: Port legacy make tasks.
- Create qecsimext repository to document and validate extensible packaging.
