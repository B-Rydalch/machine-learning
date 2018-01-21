from random import *
class Movie (object):

    def __init__(self, title, year, runtime):
        self.title = title
        self.year = year
        self.runtime = runtime
        if (self.runtime < 0):
            self.runtime = 0

    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins."

    def convert(self):
        hours = self.runtime / 60
        minutes = self.runtime % 60
        return hours, minutes


def create_movie_list():
    movie_list = []
    movie_list.append(Movie("Thor Ragnarok", 2017, 128))
    movie_list.append(Movie("Coco", 2017, 115))
    movie_list.append(Movie("Back to the Future", 1968, 93))
    movie_list.append(Movie("Star Trek", 1993, 72))
    return movie_list

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def main():
    best_ever = create_movie_list()
    ratings = {}
    for movie in best_ever:
        ratings[movie.title] = round(uniform(0.0, 5.0),1)
    for review in ratings:
        print review + " " + str(ratings[review])

    data = get_movie_data()

    print len(data)
if __name__ == "__main__":
    main()
