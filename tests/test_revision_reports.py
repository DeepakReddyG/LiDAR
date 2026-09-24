"""Report regeneration must never move blind annotation packages."""
from revision.paper_outputs import archive_existing

def test_archive_only_generated_files_and_preserve_packages(tmp_path):
    package=tmp_path/'holdout_annotation';package.mkdir();(package/'task.txt').write_text('blind input')
    audit=tmp_path/'pilot_label_audit';audit.mkdir()
    (tmp_path/'results.json').write_text('old generated result')
    (tmp_path/'user_notes.md').write_text('preserve')
    archive_existing(tmp_path, {'results.json'})
    assert (package/'task.txt').read_text()=='blind input'
    assert audit.is_dir()
    assert (tmp_path/'user_notes.md').read_text()=='preserve'
    assert not (tmp_path/'results.json').exists()
    assert len(list((tmp_path/'archive').glob('*/results.json')))==1
