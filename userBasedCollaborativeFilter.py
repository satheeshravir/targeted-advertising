__author__ = 'FarhanKhwaja'

import sys
from math import sqrt
import operator


def loadData(filename):
    users = {}
    shopVisits = {}
    matrix = [[0 for x in range(20)] for j in range(10000)]
    with open(filename, 'r') as fileData:
        next(fileData)
        for i, data in enumerate(fileData):
            users.setdefault(i, float(data.split(',')[0]))
            normData = [float(i) for i in data.split(',')[1:]]
            for j in range(len(data.split(',')[1:])):
                shopVisits.setdefault(j, 0)
                shopVisits[j] += float(data.split(',')[j])
                matrix[i][j] = normData[j]

    return users, shopVisits, matrix


def normalizeValue(visitArray, totalVisit):
    visitArray = [int(i) for i in visitArray]
    maxVal = int(totalVisit)
    minVal = min(visitArray)
    normArray = [0] * len(visitArray)

    for i, val in enumerate(visitArray):
        if (maxVal-minVal) != 0:
            normArray[i] = round(((val - minVal)/(maxVal - minVal)) * 5, 5)
        else:
            normArray[i] = 0
    return normArray


def weightSimilarity(users, shopVisits, matrix, username, n, k):

    posIdx = []
    missingValIdx = []

    unvisitedShopRatings = {}

    if n <= 0:
        return 0

    for index, rating in enumerate(matrix[username-1]):
        if rating != 0:
            posIdx.append(index)
        else:
            missingValIdx.append(index)

    #Non-Rated, Rated combination for weight
    for i in missingValIdx:
        score = [(calculateWeight(i, j, users, matrix), j) for j in posIdx]
        num = 0
        denom = 0
        newScore = []
        recommendedShops = []

        for y, x in enumerate(shopVisits.keys()):
            for sc in score:
                if sc[1] == y:
                    newScore.append((sc[0], x))

        score = sorted(newScore, key=lambda x: (-x[0], x[1]))

        for x, y in score[:n]:
            num += matrix[username-1][y] * x
            denom += x

        if denom == 0:
            rating = 0
        else:
            rating = round(num/denom, 5)

        unvisitedShopRatings.setdefault(i, 0.0)
        unvisitedShopRatings[i] = rating

    for shop, rating in sorted(unvisitedShopRatings.items(), key=operator.itemgetter(1), reverse=True):
        for m, v in shopVisits.items():
            if m == shop:
                recommendedShops.append((rating, m))

    recommendedShops.sort(key=lambda x: (-x[0], x[1]))

    reco = 0
    
    while reco != k:
        for r, m in enumerate(recommendedShops[reco]):
            if r == 0:
                rating = m
            elif r == 1:
                shop = m
        print(shop, rating)
        reco += 1


def calculateWeight(m1, m2, Users, ratingMatrix):
    ratedUser = []
    ratedUser.append([])
    ratedUser.append([])
    userRating = []
    userRating.append([])
    userRating.append([])

    for user in Users.keys():
        if ratingMatrix[user][int(m1)] != 0:
            ratedUser[0].append(user)

        if ratingMatrix[user][int(m2)] != 0:
            ratedUser[1].append(user)

    coratedUsers = set(ratedUser[0]).intersection(set(ratedUser[1]))
    if len(coratedUsers) == 0:
        return 0

    for i in coratedUsers:
        userRating[0].append(ratingMatrix[i][int(m1)])
        userRating[1].append(ratingMatrix[i][int(m2)])

    average1 = sum(userRating[0])/len(userRating[0])
    average2 = sum(userRating[1])/len(userRating[1])

    numerator = 0
    denom = {}
    denom[0] = denom[1] = 0
    for u in coratedUsers:
        numerator += (ratingMatrix[u][int(m1)] - average1) * (ratingMatrix[u][int(m2)] - average2)
        denom[0] += pow((ratingMatrix[u][int(m1)] - average1), 2)
        denom[1] += pow((ratingMatrix[u][int(m2)] - average2), 2)

    if denom[0] == 0 or denom[1] == 0:
        return 0
    else:
        return (numerator/(sqrt(denom[0])*sqrt(denom[1])))


if __name__ == "__main__":
    filename = sys.argv[1]
    username = int(sys.argv[2])
    neighbour = int(sys.argv[3])
    recommendation = int(sys.argv[4])
    users, shopVisits, matrix = loadData(filename)
    weightSimilarity(users, shopVisits, matrix, username, neighbour, recommendation)
    # print(movies['Pirates of the Caribbean: The Curse of the Black Pearl'])

    # for x,y in users.items():
    #     for i in [0,29,6]:
    #         if i == y:
    #             print(x,y)

