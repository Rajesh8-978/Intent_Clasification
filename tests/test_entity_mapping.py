import unittest

from intent_classification.entity_types import EntityType, map_label_to_entity_type


class EntityMappingTests(unittest.TestCase):
    def test_maps_business_friendly_labels_to_entity_types(self):
        self.assertEqual(
            map_label_to_entity_type("Statement of Affairs"),
            EntityType.STATEMENT_OF_AFFAIRS,
        )
        self.assertEqual(
            map_label_to_entity_type("Cheque Deposit Request"),
            EntityType.CHEQUE_DEPOSIT_REQUEST,
        )
        self.assertEqual(
            map_label_to_entity_type("General Case"),
            EntityType.CASE,
        )

    def test_rejects_unmapped_label(self):
        with self.assertRaises(ValueError):
            map_label_to_entity_type("Not a real business intent")


if __name__ == "__main__":
    unittest.main()
