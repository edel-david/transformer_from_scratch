import os, numpy as np

class Track:
    def __init__(self, hash, genre, set_name):
        self.hash = hash
        self.genre = genre
        self.set_name = set_name
        self.memmap = None

    def get_path(self, data_dir="data", suffix=".bin"):
        return os.path.join(
            data_dir,
            "tokenized",
            self.set_name,
            f"GENRE_{self.genre}",
            self.hash + suffix,
        )

    def exists(self, data_dir="data"):
        return os.path.exists(self.get_path(data_dir=data_dir))

    def get_memmap(self, data_dir="data"):
        if self.memmap is None:
            self.memmap = np.memmap(self.get_path(data_dir=data_dir), dtype=np.uint16, mode="r")
        return self.memmap


class Dataset:
    tracks: dict[str, list[Track]]

    def __init__(self, name, data_dir="data", context_length=256, uniform=False, only_genres=None):
        self.name = name
        self.sliced_tracks = None
        self.uniform = uniform

        if not os.path.exists(os.path.join(data_dir, "tokenized", name)):
            raise FileNotFoundError(f"Dataset {name} not found")

        _genres = os.listdir(os.path.join(data_dir, "tokenized", name))
        self.genres = list()

        for genre in _genres:
            if genre.startswith("GENRE_"):
                if only_genres is not None and genre[6:] not in only_genres:
                    continue
                self.genres.append(genre[6:])

        self.genres = sorted(self.genres)

        self._genres_to_idx = {}
        for i, genre in enumerate(self.genres):
            self._genres_to_idx[genre] = i

        self.tracks = {}

        for genre in self.genres:
            self.tracks[genre] = []
            for file in os.listdir(
                os.path.join(data_dir, "tokenized", name, f"GENRE_{genre}")
            ):
                if file.endswith(".bin"):
                    track = Track(file[:-4], genre, name)
                    if track.exists(data_dir=data_dir):
                        self.tracks[genre].append(track)
                    else:
                        print(f"File {file} not found")
                        #raise FileNotFoundError(f"Tokenized file {file} not found")
        
        self.get_slices(context_length=context_length, data_dir=data_dir)
        self.genre_probabilities = np.array(
            [len(self.sliced_tracks[genre]) for genre in self.sliced_tracks.keys()]
        )
        self.genre_probabilities = self.genre_probabilities / self.genre_probabilities.sum()



    def genre_to_idx(self, genre):
        return self._genres_to_idx[genre]

    def get_slices(self, context_length, data_dir="data"):
        if self.sliced_tracks is None:
            self.sliced_tracks = {}
            for genre in self.genres:
                self.sliced_tracks[genre] = []
                for track in self.tracks[genre]:
                    track = track.get_memmap(data_dir=data_dir)
                    for i in range(
                        0, len(track) - context_length, context_length // 2
                    ):  # overlap of 50%
                        self.sliced_tracks[genre].append(track[i : i + context_length])
        return self.sliced_tracks

    def get_batch_from_slices(self, batch_size, rng):

        selected_slices = []
        selected_genres = []
        if self.uniform:
            selected_genres_strings = rng.choice(
                list(self.sliced_tracks.keys()), size=(batch_size,)
            )
        else:
            selected_genres_strings = rng.choice(
                    list(self.sliced_tracks.keys()), p=self.genre_probabilities
                ,size=(batch_size,))
            
        for selected_genre in selected_genres_strings:
            selected_genres.append(self.genre_to_idx(selected_genre))
            selected_slice_idx = rng.integers(len(self.sliced_tracks[selected_genre]))
            slice = self.sliced_tracks[selected_genre][selected_slice_idx]
            slice = list(slice)
            slice[0] = 3  # 3 is the genre token
            selected_slices.append(slice)

        x = np.stack(selected_slices)
        y = np.stack(selected_genres)

        return x, y
