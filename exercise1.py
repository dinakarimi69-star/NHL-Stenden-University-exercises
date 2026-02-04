    """
    This method returns the fruit name by getting the string at a specific index of the set.

    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corrosponding to the index ``fruit_id``

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   It does not print the fruit at the correct index, why is the returned result wrong?
    The function id_to_fruit was intended to return a fruit name based on a numeric index (fruit_id) from a set of fruits. However, the function returned inconsistent and incorrect results.
``
    | ``2   How could this be fixed? To fix the issue, the set was converted into a sorted list, ensuring a deterministic and predictable order before indexing.``

    This example demonstrates the issue:
    name1, name3 and name4 are expected to correspond to the strings at the indices 1, 3, and 4:
    'orange', 'kiwi' and 'strawberry'..

    >>> name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    """
    
   def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    fruits_list = sorted(fruits)

    if fruit_id < 0 or fruit_id >= len(fruits_list):
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

    return fruits_list[fruit_id]


