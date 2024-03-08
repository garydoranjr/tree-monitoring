import os.path as op

configfile: "config/snakemake.yml"

crown_dir = op.join(config['root_dir'], config['crown_subdir'])
crown_output_dir = op.join(crown_dir, config['crown_output_subdir'])

def get_crown_ids(species):
    return set(glob_wildcards(
        op.join(crown_dir, species, '{crownid}_{yy}_{mm}_{dd}.png')
    ).crownid)

videos = sum([
    expand(
        op.join(crown_output_dir, '{species}_{crownid}.mp4'),
        species=[species], crownid=get_crown_ids(species)
    )
    for species in config['species']
], [])


rule all_videos:
    input:
        videos


rule generate_video:
    input:
        op.join(crown_dir, '{species}')
    output:
        op.join(crown_output_dir, '{species}_{crownid}.mp4')
    params:
        config['config_video']
    wildcard_constraints:
        crownid='\d+'
    shell:
        "python scripts/generate_sequence_video.py {input} {wildcards.crownid} {params} {output}"
