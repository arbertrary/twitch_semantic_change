from pathlib import Path
import jinja2

if __name__ == '__main__':
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "analysis/counts_template.yaml"

    template = templateEnv.get_template(TEMPLATE_FILE)

    dir = Path("analysis/jobs/")
    dir.mkdir(exist_ok=True)

    for timestep in [201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 202001, 202002, 202003, 202004]:
        # print(str(timestep)[:4], str(timestep)[4:])

        #        job = template.render(month=str(timestep)[:4] + "-" + str(timestep)[4:])
        job = template.render(month=str(timestep))
        dir.joinpath("job_month_%i.yml" % timestep).write_text(job)
