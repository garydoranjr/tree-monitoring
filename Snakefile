import os.path as op

configfile: "config/snakemake.yml"

crown_dir = op.join(config['root_dir'], config['crown_subdir'])
crown_output_dir = op.join(config['root_dir'], config['crown_output_subdir'])

planet_dir = op.join(config['root_dir'], config['planet_subdir'])
planet_downloads = op.join(planet_dir, config['planet_downloads_subdir'])


def get_crown_combinations():
    wild = glob_wildcards(
        op.join(crown_dir, '{genus}_{species}_{crownid}_{yy}_{mm}_{dd}.png')
    )
    unique = set(zip(wild.genus, wild.species, wild.crownid))
    genus, species, crownid = zip(*unique)
    return {
        'genus': genus,
        'species': species,
        'crownid': crownid,
    }


def get_planet_tiffs():
    wild = glob_wildcards(
        op.join(planet_downloads, '{order}', 'PSScene', '{base}_AnalyticMS_clip.tif')
    )
    unique = set(zip(wild.order, wild.base))
    order, base = zip(*unique)
    return {
        'order': order,
        'base': base,
    }


rule all_planet_rgb:
    input:
        expand(
            os.path.join(
                planet_downloads, '{order}',
                'PSScene', '{base}_AnalyticMS_clip_rgb.tif'
            ), zip, **get_planet_tiffs()
        )


rule to_rgb:
    input:
        '{base}.tif'
    output:
        '{base}_rgb.tif'
    shell:
        'gdal_translate -b 3 -b 2 -b 1 -mask "none" {input} {output} -scale -oT Byte'


rule all_videos:
    input:
        expand(
            op.join(crown_output_dir, '{genus}_{species}_{crownid}.mp4'),
            zip, **get_crown_combinations()
        )


rule generate_video:
    input:
        crown_dir
    output:
        op.join(crown_output_dir, '{genus}_{species}_{crownid}.mp4')
    params:
        config['config_video']
    wildcard_constraints:
        crownid='\d+'
    shell:
        "python scripts/generate_sequence_video.py {input} {wildcards.crownid} {params} {output}"
