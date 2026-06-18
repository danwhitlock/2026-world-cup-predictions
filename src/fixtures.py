"""2026 FIFA World Cup groups and bracket structure."""

# Team names as they appear in results.csv
GROUPS = {
    'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
    'B': ['Switzerland', 'Canada', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Scotland', 'Morocco', 'Brazil', 'Haiti'],
    'D': ['United States', 'Australia', 'Turkey', 'Paraguay'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Sweden', 'Japan', 'Netherlands', 'Tunisia'],
    'G': ['New Zealand', 'Iran', 'Belgium', 'Egypt'],
    'H': ['Uruguay', 'Saudi Arabia', 'Spain', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# All 48 teams in a flat list for quick lookup
ALL_TEAMS = [team for teams in GROUPS.values() for team in teams]

# Display names for output (where results.csv names differ from official FIFA names)
DISPLAY_NAMES = {
    'Czech Republic': 'Czechia',
    'Turkey': 'Türkiye',
    'DR Congo': 'Congo DR',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'United States': 'USA',
    'Ivory Coast': "Côte d'Ivoire",
}
