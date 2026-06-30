"""2026 FIFA World Cup groups and bracket structure."""

# Official R32 bracket structure (matches 73-88).
# Each entry is (t1_position, t2_position) where position is:
#   "1X"      → winner of group X
#   "2X"      → runner-up of group X
#   "3ABCDF"  → best qualifying 3rd-place team from those groups (Annex C resolved at runtime)
# Order: left-half matches (73-80) then right-half matches (81-88), so that
#   R16 pairs winner(73)+winner(74), winner(75)+winner(76), … in order.
# Entries are ordered so that adjacent pairs feed the correct R16/QF/SF matches.
# Adjacent pair rule: entries [0,1]→R16-M89, [2,3]→M90, [4,5]→M93, [6,7]→M94 (left half)
#                             [8,9]→M91, [10,11]→M92, [12,13]→M96, [14,15]→M95 (right half)
# QF: [M89,M90]→M97, [M93,M94]→M98 | [M91,M92]→M99, [M96,M95]→M100
# SF: [M97,M98]→M101 (left), [M99,M100]→M102 (right)
R32_BRACKET = [
    # ── Left half ──────────────────────────────────────────────────
    ("1E",  "3ABCDF"),  # M74  → pairs with M77 for R16 M89
    ("1I",  "3CDFGH"),  # M77
    ("2A",  "2B"),       # M73  → pairs with M75 for R16 M90
    ("1F",  "2C"),       # M75
    ("2K",  "2L"),       # M83  → pairs with M84 for R16 M93
    ("1H",  "2J"),       # M84
    ("1D",  "3BEFIJ"),  # M81  → pairs with M82 for R16 M94
    ("1G",  "3AEHIJ"),  # M82
    # ── Right half ─────────────────────────────────────────────────
    ("1C",  "2F"),       # M76  → pairs with M78 for R16 M91
    ("2E",  "2I"),       # M78
    ("1A",  "3CEFHI"),  # M79  → pairs with M80 for R16 M92
    ("1L",  "3EHIJK"),  # M80
    ("1J",  "2H"),       # M86  → pairs with M88 for R16 M96
    ("2D",  "2G"),       # M88
    ("1B",  "3EFGIJ"),  # M85  → pairs with M87 for R16 M95
    ("1K",  "3DEIJL"),  # M87
]

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
