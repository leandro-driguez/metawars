from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union
import random
import numpy as np

def load_internal_data() -> dict:
    with open("../data/internal_data.json", "r", encoding="utf-8") as data_fd:
        return json.load(data_fd)

DATA = load_internal_data()

UNITS_DATA = DATA["units"]
WEAPON_DATA = DATA["weapon"]
ARMOUR_DATA = DATA["armour"]
NAMES_DATA = DATA["names"]
DEATHS_DATA = DATA["deaths"]

UNITY_TYPES = list(UNITS_DATA.keys())
WEAPON_TYPES = list(list(WEAPON_DATA.values())[0].keys())
ARMOUR_TYPES = list(list(ARMOUR_DATA.values())[0].keys())
UNIT_NAMES = list(NAMES_DATA.values())
DEATHS = list(DEATHS_DATA.values())

class Weapon:
    wood = "wood"
    steel = "steel"
    diamond = "diamond"


class Armour:
    wood = "wood"
    steel = "steel"
    diamond = "diamond"


WPN_MULT = {
    Weapon.wood: 1.2,
    Weapon.steel: 1.6,
    Weapon.diamond: 2.0,
}

DEF_MULT = {
    Armour.wood: 1.2,
    Armour.steel: 1.6,
    Armour.diamond: 2.0,
}


class Unit:
    peasant = "peasant"
    swordman = "swordman"
    spearman = "spearman"
    archer = "archer"
    defender = "defender"
    horseman = "horseman"
    sniper = "sniper"
    knight = "knight"
    elefant = "elefant"

    def __init__(self, unit_type: str, weapon: str, armour: str, level: int, name=""):
        assert unit_type in UNITY_TYPES, f"Invalid unit type '{unit_type}'"
        assert weapon in WEAPON_TYPES, f"Invalid weapon type '{weapon}'"
        assert armour in ARMOUR_TYPES, f"Invalid armour type '{armour}'"
        assert level > 0, "Level must be greater than 1"

        self.unit_type = unit_type
        self.weapon = weapon
        self.armour = armour
        self.level = level
        self.name = name

        unit_data = UNITS_DATA[unit_type]
        self.weapon_cost = WEAPON_DATA[unit_type][weapon]
        self.armour_cost = ARMOUR_DATA[unit_type][weapon]
        self.unit_cost = unit_data["cost"] * (level**2)

        if(name == ""):
            self.name = UNIT_NAMES[random.randint(0, len(UNIT_NAMES) -1)]
        else:
            self.name = name

        self.cost = self.unit_cost + self.armour_cost + self.unit_cost

        self.attack = unit_data["atk"] * WPN_MULT[weapon] * (1.2**level)
        self.min_damage = unit_data["dmg"][0] * (1.2**level)
        self.max_damage = unit_data["dmg"][1] * (1.2**level)
        self.defense = unit_data["def"] * DEF_MULT[armour] * (1.2**level)
        self.hit_points = unit_data["hp"] * (1.2**level)
        self.speed = unit_data["spd"] * (1.2**level)
        self.atk_range = unit_data["ran"]

    def __repr__(self) -> str:
        return (
            f"{self.name} -> ({self.unit_type}, cost={self.cost}, atk={self.attack}, "
            f"dmg={self.min_damage}-{self.max_damage}, def={self.defense}, "
            f"hp={self.hit_points}, spd={self.speed}, rng={self.atk_range})"
        )

    def __str__(self) -> str:
        return repr(self)

    def clone(self) -> Unit:
        return Unit(
            unit_type=self.unit_type,
            weapon=self.weapon,
            armour=self.armour,
            level=self.level,
        )

    @staticmethod
    def load_army(path: Union[str, Path]) -> List[Unit]:
        path = Path(path) if isinstance(path, str) else path

        if not path.exists() or not path.is_file():
            raise ValueError("Invalid army file path")

        if path.suffix != ".json":
            raise ValueError("Invalid army file type")

        with open(path, "r", encoding="utf-8") as army_fd:
            army_data = json.load(army_fd)

        army: List[Unit] = []

        for unit in army_data:
            army.append(
                Unit(
                    unit_type=unit["unit_type"],
                    weapon=unit["weapon"],
                    armour=unit["armour"],
                    level=unit["level"],
                    name=unit["name"]
                )
            )

        return army

    @staticmethod
    def save_army(army: List[Unit], path: Union[str, Path]):
        data = [
            dict(
                unit_type=unit.unit_type,
                weapon=unit.weapon,
                armour=unit.armour,
                level=unit.level,
                name=unit.name,
            )
            for unit in army
        ]
        with open(path, "w+", encoding="utf-8") as data_fd:
            json.dump(data, data_fd, indent=4, ensure_ascii=False)


def army_cost(army: List[Unit]) -> int:
    return sum(unit.cost for unit in army)


def simulation(left_army: List[Unit], right_army: List[Unit]):
    # Seeing if the mony is OK
    print("\n\n Starting simulation")
    left_cost = army_cost(left_army)
    right_cost =  army_cost(right_army)
    if(left_cost > 100000 or right_cost > 100000):
        if (left_cost > 100000 and right_cost > 100000):
            print("Both commanders are shameless. The soldiers return home disappointed that they have not been able to cut off a single head.")
            return 0
        else:
            if(right_cost > left_cost ):
                cheater = "right"
                result = 1
            else:
                cheater = "left"
                result = 2
            print(f"The commander of the {cheater} army is a cheater. Soldiers return home disappointed that they were not able to stab anyone to death")
            return result
        
    # Lets simulate
    while(len(left_army) > 0 and len(right_army) > 0):
        print("+++++++++++++++++++++++++++++++++++++++++")
        attack_order = sorted(left_army + right_army, key=lambda u: u.speed, reverse=True)

        for unit in attack_order:
            if(len(right_army) <= 0 or len(left_army) <= 0):
                break
            if(unit.hit_points <= 0):
                continue
            
            if(unit in left_army):
                enemy = right_army
                allies = left_army
            else:
                enemy = left_army
                allies = right_army
            actual_range = unit.atk_range - (len(allies) -1 + allies.index(unit))
            if(actual_range <= 0):
                print(f"{unit.name} couldn't attack because the idiot was out of range ")
                continue

            actual_range = min(actual_range, len(enemy)-1)

            defender = enemy[random.randint(0, actual_range)]
            attacker = unit

            # Let's fight!
            target = defender.defense + 20
            throw = random.randint(0, 20)
            if(throw == 1):
                print("CRITICAL MISS!")
                print(f"{unit.name} misses")
                continue

            if(throw == 20):
                print("CRITICAL HIT!")

            if(throw + attacker.attack < target and throw != 20):
                print(f"{unit.name} misses")
                continue

            # The attack was a hit!

            damage = np.random.uniform(unit.min_damage, unit.max_damage)
            if (throw == 20):
                damage = damage*3
            print(f"{attacker.name} landed an attack on {defender.name} and dealt [{damage}] damage!")
            defender.hit_points -= damage
            if(defender.hit_points <= 0):
                print(f"{defender.name} {DEATHS[random.randint(0, len(DEATHS) - 1)]}")
                enemy.remove(defender)

        print(f"Round results: left army {len(left_army)} - {len(right_army)} right army")
        print("+++++++++++++++++++++++++++++++++++++++++")
    
    print(f"Final results: left army {len(left_army)} - {len(right_army)} right army")
    return (len(left_army), len(right_army))

