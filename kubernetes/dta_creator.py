from pathlib import Path
import jinja2
import os

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
TEMPLATE_FILE = "templates/dta_emb_template.yaml"

template = templateEnv.get_template(TEMPLATE_FILE)

dir = Path("embeddings/jobs")
dir.mkdir(exist_ok=True)

for ts in os.listdir("/home/stud/bernstetter/datasets/dta/bins_50y/"):
    if ts in ["na", "1450", "1500", "1550", "1950"]:
        continue
    print(ts)
    job = template.render(dir=ts)
    dir.joinpath("job_dir_%s.yml" % ts).write_text(job)
