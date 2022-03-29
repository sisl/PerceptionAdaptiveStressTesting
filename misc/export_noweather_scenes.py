from re import S
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm

def main():
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=dataroot)

    fog_cnt = 0
    rain_cnt = 0
    snow_cnt = 0
    non_weather_scenes = []

    weather_scenes = {'train': [], 'val':[]}
    nonweather_scene_split = {'train': [], 'val':[]}

    for i in tqdm(range(len(nusc.scene))):
        scene = nusc.scene[i]
        description = scene['description'].lower()

        if 'rain' not in description and 'fog' not in description and 'snow' not in description or 'after rain' in description:
            non_weather_scenes.append(scene['token'])
            if scene['name'] in splits.train_detect:
                nonweather_scene_split['train'].append(scene['token'])
            elif scene['name'] in splits.val:
                nonweather_scene_split['val'].append(scene['token'])
        else:
            print(description)
            if scene['name'] in splits.train_detect:
                scene_type = 'train'
            else:
                scene_type = 'val'

            weather_scenes[scene_type].append(scene['token'])

    print('Num non-weather scenes: {}'.format(len(non_weather_scenes)))
    print('Num weather detect train: {}'.format(len(weather_scenes['train'])))
    print('Num weather detect val: {}'.format(len(weather_scenes['val'])))

    scene_fname = './nonweather_scene_tokens.txt'
    with open(scene_fname, 'w') as f:
        for token in non_weather_scenes:
            f.write(token + '\n')


    train_scene_fname = './train_nonweather_scene_tokens.txt'
    with open(train_scene_fname, 'w') as f:
        for token in nonweather_scene_split['train']:
            f.write(token + '\n')


    val_scene_fname = './val_nonweather_scene_tokens.txt'
    with open(val_scene_fname, 'w') as f:
        for token in nonweather_scene_split['val']:
            f.write(token + '\n')

    #print('Nbr rain scenes: {}, nbr fog scenes: {}, nbr snow scenes: {}'.format(rain_cnt, fog_cnt, snow_cnt))



if __name__ == '__main__':
    main()