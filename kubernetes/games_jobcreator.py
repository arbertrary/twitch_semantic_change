from pathlib import Path
import jinja2
import itertools

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
TEMPLATE_FILE = "hamilton.yaml"

template = templateEnv.get_template(TEMPLATE_FILE)

dir = Path("jobs")
dir.mkdir(exist_ok=True)

models = ["vec_128_w5_mc100_iter3_sg0_lc0_clean0_w2v1", "vec_128_w5_mc50_iter5_sg0_lc0_clean0_w2v1", "vec_128_w5_mc50_iter3_sg0_lc0_clean0_w2v1"]
games = ["dota", "fortnite", "lol", "overwatch", "tarkov"]
gamescombo = list(itertools.combinations(games, 2))
print(gamescombo)

for game1,game2 in gamescombo:
    for i, m in enumerate(models):
        job = template.render(g1=game1, g2=game2, model=m, num=i)
        dir.joinpath("job_games_%s%s_%i.yml" % (game1, game2, i)).write_text(job)


# for fold in range(15):
#     job = template.render(fold_num=fold)
#     dir.joinpath("job_fold_%i.yml" % fold).write_text(job)
