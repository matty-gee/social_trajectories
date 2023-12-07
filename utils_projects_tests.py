from spatial_funcs import *


def test_calculate_circularity():
    # Test case 1
    coords = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    expected = 1.0
    actual = calculate_circularity(coords)
    assert np.isclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
    # Test case 2
    coords = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]
    expected = 1.0
    actual = calculate_circularity(coords)
    assert np.isclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
    # Test case 3
    coords = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
    expected = 1.0
    actual = calculate_circularity(coords)
    assert np.isclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
    # Test case 4
    coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    expected = 1.0
    actual = calculate_circularity(coords)
    assert np.isclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'

test_calculate_circularity()


def test_calc_polar_coords():
    # Test case 1
    coords = [[1, 2], [3, 4], [5, 6]]
    center = [0, 0]
    expected = [[2.23606798, 0.46364761],
                [4.47213595, 0.89605538],
                [6.70820393, 1.32746071]]
    actual = calc_polar_coords(coords, center)
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
    # Test case 2
    coords = [[0, 0], [1, 1], [2, 2]]
    center = [1, 1]
    expected = [[0.70710678, 0.78539816],
                [1.41421356, 1.57079633],
                [2.12132034, 2.35619449]]
    actual = calc_polar_coords(coords, center)
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
    # Test case 3
    coords = [[0, 0], [1, 0], [0, 1]]
    center = [0, 0]
    expected = [[0., 0.],
                [1., 0.],
                [1., 1.57079633]]
    actual = calc_polar_coords(coords, center)
    assert np.allclose(actual, expected, rtol=1e-05, atol=1e-08), f'Error: expected {expected}, got {actual}'
    
test_calc_polar_coords()


def test_cart_to_cylindrical():
    x, y, z = 1, 1, 1
    r, theta, z_new = cart_to_cylindrical(x, y, z)
    assert np.isclose(r, np.sqrt(2)), f'r is not correct: should be {np.sqrt(2)}, but is {r}'
    assert np.isclose(theta, np.pi/4), f'theta is not correct: should be {np.pi/4}, but is {theta}'
    assert z == z_new, f'z is not correct: should be {z}, but is {z_new}'

    x, y, z = 1, 0, 0
    r, theta, z_new = cart_to_cylindrical(x, y, z)
    assert r == 1, f'r is not correct: should be 1, but is {r}'
    assert theta == 0, f'theta is not correct: should be 0, but is {theta}'
    assert z == z_new

    x, y, z = 0, 1, 0
    r, theta, z_new = cart_to_cylindrical(x, y, z)
    assert r == 1, f'r is not correct: should be 1, but is {r}'
    assert theta == np.pi/2, f'theta is not correct: should be {np.pi/2}, but is {theta}'
    assert z == z_new, f'z is not correct: should be {z}, but is {z_new}'

    x, y, z = 0, 0, 1
    r, theta, z_new = cart_to_cylindrical(x, y, z)
    assert r == 0, f'r is not correct: should be 0, but is {r}'
    assert theta == 0, f'theta is not correct: should be 0, but is {theta}'
    assert z == z_new, f'z is not correct: should be {z}, but is {z_new}'

test_cart_to_cylindrical()


# this might be incorrect now...
def test_cart_to_spherical():
    # Test conversion of Cartesian coordinates to spherical coordinates
    x = 1
    y = 0
    z = 0
    r, theta, phi = cart_to_spherical(x, y, z)
    assert abs(r - 1) < 1e-6, f'radius is {r} and should be 1'
    assert abs(theta - radians(0)) < 1e-6, f'theta is {theta} and should be 0'
    assert abs(phi - radians(90)) < 1e-6, f'phi is {phi} and should be 90'

    x = 0
    y = 1
    z = 0
    r, theta, phi = cart_to_spherical(x, y, z)
    assert abs(r - 1) < 1e-6, f'radius is {r} and should be 1'
    assert abs(theta - radians(90)) < 1e-6, f'theta is {theta} and should be 90'
    assert abs(phi - radians(90)) < 1e-6, f'phi is {phi} and should be 90'

    x = 0
    y = 0
    z = 1
    r, theta, phi = cart_to_spherical(x, y, z)
    assert abs(r - 1) < 1e-6, f'radius is {r} and should be 1'
    assert abs(theta - radians(0)) < 1e-6, f'theta is {theta} and should be 0'
    assert abs(phi - radians(0)) < 1e-6, f'phi is {phi} and should be 0'

test_cart_to_spherical()


def test_latlong_to_cart():
    # Test conversion of latitude and longitude to Cartesian coordinates
    lat = radians(0)
    lon = radians(0)
    x, y, z = latlong_to_cart(lat, lon)
    assert abs(x - 1) < 1e-6
    assert abs(y - 0) < 1e-6
    assert abs(z - 0) < 1e-6

    lat = radians(0)
    lon = radians(90)
    x, y, z = latlong_to_cart(lat, lon)
    assert abs(x - 0) < 1e-6
    assert abs(y - 1) < 1e-6
    assert abs(z - 0) < 1e-6

    lat = radians(90)
    lon = radians(0)
    x, y, z = latlong_to_cart(lat, lon)
    assert abs(x - 0) < 1e-6
    assert abs(y - 0) < 1e-6
    assert abs(z - 1) < 1e-6

    # Test conversion of multiple points
    lat = np.array([radians(0), radians(0), radians(90)])
    lon = np.array([radians(0), radians(90), radians(0)])
    x, y, z = latlong_to_cart(lat, lon)
    assert abs(x - np.array([1, 0, 0])).all() < 1e-6
    assert abs(y - np.array([0, 1, 0])).all() < 1e-6
    assert abs(z - np.array([0, 0, 1])).all() < 1e-6

test_latlong_to_cart()


def test_cart_to_latlong():
    # Test conversion of Cartesian coordinates to latitude and longitude
    x = 1
    y = 0
    z = 0
    lat, lon = cart_to_latlong(x, y, z)
    # assert that the values are close with np.isclose
    assert np.isclose(lat, radians(0), atol=1e-6), f'lat = {lat}'
    assert np.isclose(lon, radians(0), atol=1e-6), f'lon = {lon}'

    x = 0
    y = 1
    z = 0
    lat, lon = cart_to_latlong(x, y, z)
    assert np.isclose(lat, radians(0), atol=1e-6), f'lat = {lat}'
    assert np.isclose(lon, radians(90), atol=1e-6), f'lon = {lon}'

    x = 0
    y = 0
    z = 1
    lat, lon = cart_to_latlong(x, y, z)
    assert np.isclose(lat, radians(90), atol=1e-6), f'lat = {lat}'
    assert np.isclose(lon, radians(0), atol=1e-6), f'lon = {lon}'

test_cart_to_latlong()


def test_geo_dist():
    # Test distance between two points with same coordinates
    x = np.array([[1, 0, 0]])
    y = np.array([[1, 0, 0]])
    assert abs(geo_dist(x, y) - 0).all() < 1e-6

    # Test distance between two points with different coordinates
    x = np.array([[1, 0, 0]])
    y = np.array([[0, 1, 0]])
    assert abs(geo_dist(x, y) - radians(90)).all() < 1e-6

    # Test distance between two points with opposite coordinates
    x = np.array([[1, 0, 0]])
    y = np.array([[-1, 0, 0]])
    assert abs(geo_dist(x, y) - radians(180)).all() < 1e-6

    # Test distance between multiple points
    x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert abs(geo_dist(x, y) - radians(90)).all() < 1e-6

test_geo_dist()


def test_xy_to_circle_circum():
    # Test first quadrant
    x, y = 1, 1
    expected_projection = [math.sqrt(2)/2, math.sqrt(2)/2]
    expected_distance = math.sqrt(2) - 1
    projection, distance = project_xy_to_circle_circum(x, y)
    assert np.allclose(projection, expected_projection), f"Projection is incorrect: expected={expected_projection}, got={projection}"
    assert np.allclose(distance, expected_distance), f"Distance is incorrect: expected={expected_distance}, got={distance}"
    
    # Test second quadrant
    x, y = -1, 1
    expected_projection = [-math.sqrt(2)/2, math.sqrt(2)/2]
    expected_distance = math.sqrt(2) - 1
    projection, distance = project_xy_to_circle_circum(x, y)
    assert np.allclose(projection, expected_projection), f"Projection is incorrect: expected={expected_projection}, got={projection}"
    assert np.allclose(distance, expected_distance), f"Distance is incorrect: expected={expected_distance}, got={distance}"

    
    # Test third quadrant
    x, y = -1, -1
    expected_projection = [-math.sqrt(2)/2, -math.sqrt(2)/2]
    expected_distance = math.sqrt(2) - 1
    projection, distance = project_xy_to_circle_circum(x, y)
    assert np.allclose(projection, expected_projection), f"Projection is incorrect: expected={expected_projection}, got={projection}"
    assert np.allclose(distance, expected_distance), f"Distance is incorrect: expected={expected_distance}, got={distance}"
    
    # Test fourth quadrant
    x, y = 1, -1
    expected_projection = [math.sqrt(2)/2, -math.sqrt(2)/2]
    expected_distance = math.sqrt(2) - 1
    projection, distance = project_xy_to_circle_circum(x, y)
    assert np.allclose(projection, expected_projection), f"Projection is incorrect: expected={expected_projection}, got={projection}"
    assert np.allclose(distance, expected_distance), f"Distance is incorrect: expected={expected_distance}, got={distance}"
    
    # Test origin
    x, y = 0, 0
    expected_projection = [0, 0]
    expected_distance = 0
    projection, distance = project_xy_to_circle_circum(x, y)
    assert np.allclose(projection, expected_projection), f"Projection is incorrect: expected={expected_projection}, got={projection}"
    assert np.allclose(distance, expected_distance), f"Distance is incorrect: expected={expected_distance}, got={distance}"
    
test_xy_to_circle_circum()