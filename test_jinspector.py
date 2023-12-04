from jinspector import JPropT, InspectorCtx, JSONPathParser, QueryExec
from jinspector import jpath_to_string, ELogLev, set_log_lev
from jinspector import read_up_to_ext
from jinspector import Expression, EExprType, Operator, EOpType
import unittest

JSON1 = '''
{
  "store": {
    "book": [
      {
        "category": "reference",
        "author": "Nigel Rees",
        "title": "Sayings of the Century",
        "price": 8.95
      },
      {
        "category": "fiction",
        "author": "Herman Melville",
        "title": "Moby Dick",
        "isbn": "0-553-21311-3",
        "price": 8.99
      },
      {
        "category": "fiction",
        "author": "J.R.R. Tolkien",
        "title": "The Lord of the Rings",
        "isbn": "0-395-19395-8",
        "price": 22.99
      }
    ],
    "bicycle": {
      "color": "red",
      "price": 19.95
    }
  },
  "expensive": 10
}
'''

JSON2 = '''
{
  "name": "Rose Kolodny",
  "phoneNumbers": [
    {
      "type": "home",
      "number": "954-555-1234"
    },
    {
      "type": "work",
      "number": "754-555-5678"
    }
  ]
}
'''

JSON3 = '''
{
    "firstName": "John",
    "lastName": "doe",
    "age": 26,
    "address": {
        "streetAddress": "naist street",
        "city": "Nara",
        "postalCode": "630-0192"
    },
    "phoneNumbers": [
        {
            "type": "iPhone",
            "number": "0123-4567-8888"
        },
        {
            "type": "home",
            "number": "0123-4567-8910"
        }
    ],
    "a" : {
        "arr": [1,2,3,4],
        "obj": {
            "i": 10,
            "arr": ["11", "22", "33"],
            "arr2": [ { "p": 1}, {"p": 2 }, {"r": 40}
                ]
        }
    }
}
'''


class SchemaTests(unittest.TestCase):
    def test_basic(self):
        ctx = InspectorCtx()
        ctx.load_from_str(JSON2)
        ok, msg = ctx.schema_tree()
        self.assertTrue(ok)
        expected_schema = '''$: <Object> {
  name: <str>
  phoneNumbers: <Array> [
    Array of: <Object> {
      type: <str>
      number: <str>
    }
  ]
}'''
        self.assertEqual(msg, expected_schema)


class JPATHTests(unittest.TestCase):
    def test_no_errors(self):
        test_cases = [
            '$..*',
            '$.store.book[0].title',
            ("$['store']['book'][0]['title']", '$.store.book[0].title'),
            ("$['store'].book[0].title", '$.store.book[0].title'),
            '$.foo.bar',
            ('foo.bar', '$.foo.bar'),
            '$[0].status',
            ('[0].status', '$[0].status'),
            '$.store.*',
            '$.store.bicycle.color',
            '$.store..price',
            '$..price',
            '$.store.book[*]',
            '$..book[*]',
            '$..book[*].title',
            '$..book[0]',
            '$..book[0].title',
            '$..book[0,1].title',
            '$..book[:2].title',
            '$..book[-1:].title',
            '$..book[(@.length-1)].title',
            "$..book[?(@.author=='J.R.R. Tolkien')].title",
            '$..book[?(@.isbn)]',
            '$..book[?(!@.isbn)]',
            '$..book[?(@.price < 10)]',
            '$..book[?(@.price > $.expensive)]',
            '$..book[?(@.author =~ /.*Tolkien/i)]',
            "$..book[?(@.category == 'fiction' || @.category == 'reference')]",
            "$..book[?(@.category in ['fiction', 'reference'])].name",
        ]

        for tc in test_cases:
            exp = tc if type(tc) is str else tc[1]
            tc = tc if type(tc) is str else tc[0]

            parser = JSONPathParser(tc)
            props, err = parser.parse()
            if len(err) != 0:
                prop = ''
                for p in props:
                    prop += str(p)
                    prop += '|'
                msg = f"Error while parsing {tc}:\n {err}\nobj: {prop}"
                self.assertTrue(False, msg)
            self.assertEqual(exp, jpath_to_string(props))

    def test_the_same(self):
        test_cases = [
            [
                '$.store.book[0].title',
                "$['store']['book'][0]['title']",
                "$['store'].book[0].title"
            ],
            [
                '$.foo.bar',
                'foo.bar'
            ],
            [
                '$[0].status',
                '[0].status'
            ]]

        for tc in test_cases:
            jp = JSONPathParser(tc[0])
            jp = jp.parse()
            for t2 in tc[1:]:
                jp2 = JSONPathParser(t2)
                jp2 = jp2.parse()
                msg = f'{tc[0]} != {t2}'
                self.assertEqual(jp, jp2, msg)

    def test_specific(self):
        tcs = []
        to_test = "$['store']['book'][0]['title']"
        expected = [
            ('store', JPropT.NORMAL),
            ('book', JPropT.NORMAL),
            ('0', JPropT.INDEX),
            ('title', JPropT.NORMAL),
        ]
        tcs.append((to_test, expected))

        to_test = "foo.bar"
        expected = [
            ('foo', JPropT.NORMAL),
            ('bar', JPropT.NORMAL),
        ]
        tcs.append((to_test, expected))

        to_test = "a.arr[1]"
        expected = [
            ('a', JPropT.NORMAL),
            ('arr', JPropT.NORMAL),
            ('1', JPropT.INDEX),
        ]
        tcs.append((to_test, expected))

        to_test = '$.store..price'
        expected = [
            ('store', JPropT.NORMAL),
            ('', JPropT.RECURSIVE_DESCENT),
            ('price', JPropT.NORMAL),
        ]
        tcs.append((to_test, expected))

        for tc in tcs:
            to_test = tc[0]
            expected = tc[1]
            jp = JSONPathParser(to_test)
            props, err = jp.parse()
            msg = f'Testing {to_test}'
            self.assertEqual(err, '', msg)
            msg2 = f'{to_test}: len({expected}) != len({props})'
            self.assertEqual(len(expected), len(props), msg2)
            for exp, curr in zip(expected, props):
                self.assertEqual(exp[0], curr.name, msg)
                self.assertEqual(exp[1], curr.ptype, msg)


class TestParsingUtilities(unittest.TestCase):
    def test_read_up_to_ext(self):
        tcs = [
            (']', 0, ']', '', '', ''),
            ('abc]adb', 0, ']', '', '', 'abc'),
            ("?(@.color=='red')]ababab", 0, ']', "'", '', "?(@.color=='red')"),
            ("?(@.color=='red')]", 0, ']', "'", '', "?(@.color=='red')"),
            ("?(@.size in [])]abab", 0, ']',
             "'", '[', "?(@.size in [])"),
            ("?(@.size in [''])]['xxx']", 0, ']',
             "'", '[', "?(@.size in [''])"),
            ("?(@.size in ['M', 'L'])]abab", 0, ']',
             "'", '[', "?(@.size in ['M', 'L'])"),
        ]

        for tc in tcs:
            ret_str, _, _ = read_up_to_ext(tc[0], tc[1], tc[2], tc[3], tc[4])
            self.assertEqual(ret_str, tc[5])


class TestIndexParsing(unittest.TestCase):
    def test_simple_cases(self):
        tcs = [
            ('*', [1, 2, 3, 4, 10], []),
            ('10', [10], [0, 9, 11, 15]),
            ('1,2,5', [1, 2, 5], [0, 3, 4, 6]),
            ('3:5', [3, 4], [0, 1, 2, 5, 6]),
            ('3:', [3, 4, 5, 6], [0, 1, 2]),
            (':3', [0, 1, 2], [3.4, 5])
        ]

        for s, yes, no in tcs:
            p = JSONPathParser('not.relevant.here')
            prop = p.parse_expr(s)
            for i in yes:
                self.assertTrue(prop.index_match(i))
            for i in no:
                self.assertTrue(not prop.index_match(i))


class TestExpressionParsing(unittest.TestCase):
    def test_exprs(self):
        tcs = [
            ("?(@.color=='red')", [
                Expression(EExprType.PROP_NAME, 'color'),
                Expression(EExprType.OP, Operator(EOpType.EQ)),
                Expression(EExprType.VAL, 'red')]),
            ("?(@.color != 10)", [
                Expression(EExprType.PROP_NAME, 'color'),
                Expression(EExprType.OP, Operator(EOpType.NE)),
                Expression(EExprType.VAL, 10)]),
            ("?(@.color> 12.6)", [
                Expression(EExprType.PROP_NAME, 'color'),
                Expression(EExprType.OP, Operator(EOpType.GT)),
                Expression(EExprType.VAL, 12.6)]),
            ("?(@.description =~ /cat.*/i)", [
                Expression(EExprType.PROP_NAME, 'description'),
                Expression(EExprType.OP, Operator(EOpType.REG)),
                Expression(EExprType.VAL, '/cat.*/i')
            ]),
            ("?(!@.isbn)", [
                Expression(EExprType.OP, Operator(EOpType.NEG)),
                Expression(EExprType.PROP_NAME, 'isbn')
            ]),
            ("?(@.category=='fiction' && @.price < -10)", [
                Expression(EExprType.PROP_NAME, 'category'),
                Expression(EExprType.OP, Operator(EOpType.EQ)),
                Expression(EExprType.VAL, 'fiction'),
                Expression(EExprType.OP, Operator(EOpType.AND)),
                Expression(EExprType.PROP_NAME, 'price'),
                Expression(EExprType.OP, Operator(EOpType.LESS)),
                Expression(EExprType.VAL, -10),
            ]),
            ("?(@.size in ['M', 'L'])", [
                Expression(EExprType.PROP_NAME, 'size'),
                Expression(EExprType.OP, Operator(EOpType.IN)),
                Expression(EExprType.VAL_ARR, ['M', 'L']),
            ]),
            ("?('S' nin @.sizes)", [
                Expression(EExprType.VAL, 'S'),
                Expression(EExprType.OP, Operator(EOpType.NIN)),
                Expression(EExprType.PROP_NAME, 'sizes'),
            ]),
            ("?(@.name size 4)", [
                Expression(EExprType.PROP_NAME, 'name'),
                Expression(EExprType.OP, Operator(EOpType.SIZE)),
                Expression(EExprType.VAL, 4),
            ]),
            ("?(@.name empty false)", [
                Expression(EExprType.PROP_NAME, 'name'),
                Expression(EExprType.OP, Operator(EOpType.EMPTYF)),
            ]),

            ("?(@.category=='fiction' && @.price < 10)", [
                Expression(EExprType.PROP_NAME, 'category'),
                Expression(EExprType.OP, Operator(EOpType.EQ)),
                Expression(EExprType.VAL, 'fiction'),
                Expression(EExprType.OP, Operator(EOpType.AND)),
                Expression(EExprType.PROP_NAME, 'price'),
                Expression(EExprType.OP, Operator(EOpType.LESS)),
                Expression(EExprType.VAL, 10),
            ]),
        ]

        for ex, ret in tcs:
            p = JSONPathParser('not.relevant.here')
            res = p.parse_expr(ex)
            self.assertEqual(res.expr_arr, ret, f'parsing: {ex}')


class TestJPathMatching(unittest.TestCase):
    def prepare_QE(self, json, query):
        ctx = InspectorCtx()
        ctx.load_from_str(json)

        p = JSONPathParser(query)
        props, err = p.parse()
        self.assertTrue(not err)

        q = QueryExec(ctx.curr_data, props)
        return q

    def test_simple_case(self):
        q = self.prepare_QE(JSON3, 'firstName')
        q.build_up()

        self.assertEqual(q.log, ['John'])

    def test_multiple_prop(self):
        q = self.prepare_QE(JSON3, 'a.obj.i')
        q.build_up()

        self.assertEqual(q.log, [10])

    def test_with_obj(self):
        q = self.prepare_QE(JSON3, 'address')
        q.build_up()

        self.assertEqual(q.log, [
                         {'streetAddress': 'naist street', 'city': 'Nara',
                          'postalCode': '630-0192'}])

    def test_simple_index(self):
        q = self.prepare_QE(JSON3, 'a.arr[2]')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, [3])

    def test_array_with_obj(self):
        q = self.prepare_QE(JSON2, '$.phoneNumbers[:1].type')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, ['home'])

    def test_wildcard_tricky(self):
        q = self.prepare_QE(JSON2, '$.*[*].type')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, ['home', 'work'])

    def test_wildcard_middle(self):
        q = self.prepare_QE(JSON3, '$.a.*.*[1]')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, ['22', {'p': 2}])

    def test_recursive_descent_simple(self):
        q = self.prepare_QE(JSON2, '$.phoneNumbers..type')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, ['home', 'work'])

    def test_recursive_descent_matching(self):
        q = self.prepare_QE(JSON3, '$..arr')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, [
            [
                1,
                2,
                3,
                4
            ],
            [
                "11",
                "22",
                "33"
            ]
        ])

    def test_recursive_descent_with_wildcard(self):
        q = self.prepare_QE(JSON2, '$.phoneNumbers..*')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()

        self.assertEqual(q.log, [
            {
                "type": "home",
                "number": "954-555-1234"
            },
            "home",
            "954-555-1234",
            {
                "type": "work",
                "number": "754-555-5678"
            },
            "work",
            "754-555-5678"
        ])

    def test_recursive_descent_tricky(self):
        q = self.prepare_QE(JSON3, '$..a..arr.*')
        # set_log_lev(ELogLev.TRACE)
        q.build_up()
        # set_log_lev(ELogLev.NONE)

        self.assertEqual(q.log, [
            1,
            2,
            3,
            4,
            "11",
            "22",
            "33"
        ])


class TestExprMatching(unittest.TestCase):
    def prepare_QE(self, json, query):
        ctx = InspectorCtx()
        ctx.load_from_str(json)

        p = JSONPathParser(query)
        props, err = p.parse()
        self.assertTrue(not err)

        q = QueryExec(ctx.curr_data, props)
        return q

    def test_simple_case(self):
        q = self.prepare_QE(JSON1, '$.store.book[?(@.price < 10)]')
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95
            },
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            }
        ])

    def test_simple_case_eq(self):
        q = self.prepare_QE(JSON1, "$.store.book[?(@.title == 'Moby Dick')]")
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            }
        ])

    def test_regexp(self):
        q = self.prepare_QE(JSON1, '$.store.book[?(@.title =~ Mob\\w+)]')
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            }
        ])

    def test_neg(self):
        q = self.prepare_QE(JSON1, '$.store.book[?(!@.isbn)]')
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95
            }
        ])

    def test_exist(self):
        q = self.prepare_QE(JSON1, '$.store.book[?(@.isbn)]')
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            },
            {
                "category": "fiction",
                "author": "J.R.R. Tolkien",
                "title": "The Lord of the Rings",
                "isbn": "0-395-19395-8",
                "price": 22.99
            }
        ])

    def test_and(self):
        q = self.prepare_QE(JSON1, ("$.store.book[?(@.category=='fiction'"
                                    "&& @.price < 10)]"))
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            },
        ])

    def test_in(self):
        q = self.prepare_QE(
            JSON1, "$.store.book[?(@.category in ['K', 'reference', 'l])")
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95
            },
        ])

    def test_empty(self):
        q = self.prepare_QE(
            JSON1, "$.store.book[?(@.title empty false)]")
        q.build_up()
        self.assertEqual(q.log, [
            {
                "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": 8.95
            },
            {
                "category": "fiction",
                "author": "Herman Melville",
                "title": "Moby Dick",
                "isbn": "0-553-21311-3",
                "price": 8.99
            },
            {
                "category": "fiction",
                "author": "J.R.R. Tolkien",
                "title": "The Lord of the Rings",
                "isbn": "0-395-19395-8",
                "price": 22.99
            }
        ])


if __name__ == "__main__":
    unittest.main()
