"""The job's resolved options and the check that it read what k4opt gave it."""

from __future__ import annotations

from pathlib import Path

from k4opt.k4run import properties, verify

# lines as Gaudi's JobOptionsSvc writes them (Key4hep nightly 2026-10-20)
DUMP = """\
AlgExecStateSvc.AuditFinalize = False;
ApplicationMgr.EvtMax = -1;
GeoSvc.OutputLevel = 3;
GeoSvc.detectors = [ '/run/geometry/CLD_o2_v09.xml' ];
IOSvc.Input = [ '/run/samples/g/simulation/events.edm4hep.root' ];
IOSvc.Output = '/run/samples/g/reconstruction/events.edm4hep.root';
MyAlg.Tool = PrivateToolHandle('Some/Tool');
"""
COMPACT = Path("/run/geometry/CLD_o2_v09.xml")
SIM = Path("/run/samples/g/simulation/events.edm4hep.root")
RECO = Path("/run/samples/g/reconstruction/events.edm4hep.root")


def test_the_dump_is_read_as_python_literals_and_text_where_it_is_none() -> None:
    found = properties(DUMP)
    assert found["ApplicationMgr.EvtMax"] == -1
    assert found["GeoSvc.detectors"] == [str(COMPACT)]
    assert found["AlgExecStateSvc.AuditFinalize"] is False
    assert found["MyAlg.Tool"] == "PrivateToolHandle('Some/Tool')"


def test_a_job_that_read_what_k4opt_gave_it_passes() -> None:
    assert verify(DUMP, compact=COMPACT, input_file=SIM, output=RECO) is None


def test_a_job_that_read_another_geometry_says_which_and_how_to_name_its_argument() -> None:
    dump = DUMP.replace(str(COMPACT), "/k4geo/CLD_o2_v07.xml")
    message = verify(dump, compact=COMPACT, input_file=SIM, output=RECO)
    assert message is not None
    assert "GeoSvc.detectors is ['/k4geo/CLD_o2_v07.xml'], not the candidate's geometry" in message
    assert "name it with geometry:" in message
    named = verify(dump, compact=COMPACT, input_file=SIM, output=RECO, geometry="outputBasename")
    assert named is not None and "that k4opt passed as --outputBasename" in named


def test_a_dump_without_the_services_or_with_other_files_fails_for_each() -> None:
    message = verify("ApplicationMgr.EvtMax = -1;\n", compact=COMPACT, input_file=SIM, output=RECO)
    assert message is not None
    assert "GeoSvc.detectors is nothing" in message
    assert "IOSvc.Input is nothing" in message
    assert "IOSvc.Output is nothing" in message
    two = DUMP.replace(f"[ '{SIM}' ]", f"[ '{SIM}', '/other.root' ]")
    message = verify(two, compact=COMPACT, input_file=SIM, output=RECO, input_argument="inputFiles")
    assert message is not None and "that k4opt passed as --inputFiles" in message


def test_the_keys_of_maps_must_be_held_as_they_were_set() -> None:
    dump = DUMP + "Pandora.Parameters = { 'Cut' : [ '2.0' ] , 'Other' : [ 'x' ] };\n"
    common = {"compact": COMPACT, "input_file": SIM, "output": RECO}
    key = ("Pandora.Parameters", "Cut")
    assert verify(dump, keys={key: ("2.0",)}, **common) is None
    said = verify(dump, keys={key: ("0.5",)}, **common)
    assert said is not None and "Pandora.Parameters[Cut] is ['2.0'], not ['0.5']" in said
    said = verify(dump, keys={("Pandora.Parameters", "Gone"): ("1",)}, **common)
    assert said is not None and "Pandora.Parameters[Gone] is nothing" in said
    said = verify(DUMP, keys={key: ("2.0",)}, **common)
    assert said is not None and "Pandora.Parameters[Cut] is nothing" in said


def test_the_seed_must_be_held_by_the_random_engine_and_the_unique_id_service() -> None:
    common = {"compact": COMPACT, "input_file": SIM, "output": RECO}
    # as Gaudi dumps them: a property no component read says so after it
    dump = DUMP + "RndmGenSvc.Engine.Seeds = [ 7 ];\nUniqueIDGenSvc.Seed = 7; // unused\n"
    assert properties(dump)["UniqueIDGenSvc.Seed"] == 7
    assert properties("IOSvc.Input = [ 'root://eos//a.root' ];")["IOSvc.Input"] == [
        "root://eos//a.root"  # a // in a text is no comment
    ]
    assert verify(dump, seed=7, **common) is None
    said = verify(dump, seed=8, **common)
    assert said is not None
    assert "RndmGenSvc.Engine.Seeds is [7], not the stage's seed [8]" in said
    assert "UniqueIDGenSvc.Seed is 7, not the stage's seed 8" in said
    assert verify(DUMP, **common) is None  # without a seed, the job's own are not checked
