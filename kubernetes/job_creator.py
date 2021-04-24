from pathlib import Path
import jinja2

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
TEMPLATE_FILE = "job_template.yaml"

template = templateEnv.get_template(TEMPLATE_FILE)

dir = Path("jobs")
dir.mkdir(exist_ok=True)

for fold in range(15):
    job = template.render(fold_num=fold)
    dir.joinpath("job_fold_%i.yml" % fold).write_text(job)
