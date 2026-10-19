//! Minimal protobuf wire encoder and the ONNX messages the exporter needs.
//! Field numbers follow onnx.proto3 (IR version 8).

pub(crate) struct Msg(Vec<u8>);

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

impl Msg {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }

    fn tag(&mut self, field: u32, wire: u8) {
        put_varint(&mut self.0, ((field as u64) << 3) | wire as u64);
    }

    pub(crate) fn int(&mut self, field: u32, v: i64) -> &mut Self {
        self.tag(field, 0);
        put_varint(&mut self.0, v as u64);
        self
    }

    pub(crate) fn float(&mut self, field: u32, v: f32) -> &mut Self {
        self.tag(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub(crate) fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.tag(field, 2);
        put_varint(&mut self.0, b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }

    pub(crate) fn string(&mut self, field: u32, s: &str) -> &mut Self {
        self.bytes(field, s.as_bytes())
    }

    pub(crate) fn msg(&mut self, field: u32, m: &Msg) -> &mut Self {
        self.bytes(field, &m.0)
    }

    pub(crate) fn packed_ints(&mut self, field: u32, vs: &[i64]) -> &mut Self {
        let mut body = Vec::new();
        for v in vs {
            put_varint(&mut body, *v as u64);
        }
        self.bytes(field, &body)
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.0
    }
}

const FLOAT: i64 = 1;
const INT64: i64 = 7;

pub(crate) fn tensor_f32(name: &str, dims: &[i64], data: &[f32]) -> Msg {
    let mut raw = Vec::with_capacity(data.len() * 4);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut m = Msg::new();
    m.packed_ints(1, dims)
        .int(2, FLOAT)
        .string(8, name)
        .bytes(9, &raw);
    m
}

pub(crate) fn tensor_i64(name: &str, dims: &[i64], data: &[i64]) -> Msg {
    let mut raw = Vec::with_capacity(data.len() * 8);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut m = Msg::new();
    m.packed_ints(1, dims)
        .int(2, INT64)
        .string(8, name)
        .bytes(9, &raw);
    m
}

pub(crate) fn attr_int(name: &str, v: i64) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).int(3, v).int(20, 2);
    m
}

pub(crate) fn attr_float(name: &str, v: f32) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).float(2, v).int(20, 1);
    m
}

pub(crate) fn attr_str(name: &str, v: &str) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).string(4, v).int(20, 3);
    m
}

pub(crate) fn attr_ints(name: &str, vs: &[i64]) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).packed_ints(8, vs).int(20, 7);
    m
}

pub(crate) fn node(
    op: &str,
    name: &str,
    inputs: &[String],
    outputs: &[String],
    attrs: Vec<Msg>,
) -> Msg {
    let mut m = Msg::new();
    for i in inputs {
        m.string(1, i);
    }
    for o in outputs {
        m.string(2, o);
    }
    m.string(3, name).string(4, op);
    for a in &attrs {
        m.msg(5, a);
    }
    m
}

pub(crate) enum Dim {
    Fixed(i64),
    Param(&'static str),
}

pub(crate) fn value_info(name: &str, dims: &[Dim]) -> Msg {
    let mut shape = Msg::new();
    for d in dims {
        let mut dim = Msg::new();
        match d {
            Dim::Fixed(v) => dim.int(1, *v),
            Dim::Param(p) => dim.string(2, p),
        };
        shape.msg(1, &dim);
    }
    let mut tensor = Msg::new();
    tensor.int(1, FLOAT).msg(2, &shape);
    let mut ty = Msg::new();
    ty.msg(1, &tensor);
    let mut m = Msg::new();
    m.string(1, name).msg(2, &ty);
    m
}

pub(crate) fn model(
    opset: i64,
    graph_name: &str,
    nodes: &[Msg],
    initializers: &[Msg],
    inputs: &[Msg],
    outputs: &[Msg],
) -> Vec<u8> {
    let mut graph = Msg::new();
    for n in nodes {
        graph.msg(1, n);
    }
    graph.string(2, graph_name);
    for t in initializers {
        graph.msg(5, t);
    }
    for i in inputs {
        graph.msg(11, i);
    }
    for o in outputs {
        graph.msg(12, o);
    }
    let mut opset_import = Msg::new();
    opset_import.string(1, "").int(2, opset);
    let mut m = Msg::new();
    m.int(1, 8)
        .string(2, "darknet-rs")
        .msg(8, &opset_import)
        .msg(7, &graph);
    m.finish()
}

#[cfg(test)]
pub(crate) mod testing {
    pub(crate) enum Wire {
        Varint(u64),
        Len(Vec<u8>),
        Fixed32,
    }

    pub(crate) fn decode(mut b: &[u8]) -> Vec<(u32, Wire)> {
        fn varint(b: &mut &[u8]) -> u64 {
            let (mut v, mut shift) = (0u64, 0);
            loop {
                let byte = b[0];
                *b = &b[1..];
                v |= ((byte & 0x7f) as u64) << shift;
                if byte & 0x80 == 0 {
                    return v;
                }
                shift += 7;
            }
        }
        let mut out = Vec::new();
        while !b.is_empty() {
            let key = varint(&mut b);
            let field = (key >> 3) as u32;
            match key & 7 {
                0 => out.push((field, Wire::Varint(varint(&mut b)))),
                2 => {
                    let n = varint(&mut b) as usize;
                    out.push((field, Wire::Len(b[..n].to_vec())));
                    b = &b[n..];
                }
                5 => {
                    out.push((
                        field,
                        Wire::Fixed32,
                    ));
                    b = &b[4..];
                }
                other => panic!("unsupported wire type {other}"),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{Wire, decode};
    use super::*;

    #[test]
    fn varint_matches_protobuf_documentation_example() {
        // https://protobuf.dev/programming-guides/encoding/ : field 1 = 150 -> 08 96 01
        let mut m = Msg::new();
        m.int(1, 150);
        assert_eq!(m.finish(), vec![0x08, 0x96, 0x01]);
    }

    #[test]
    fn negative_int_uses_ten_byte_twos_complement() {
        let mut m = Msg::new();
        m.int(1, -1);
        let bytes = m.finish();
        assert_eq!(bytes.len(), 1 + 10);
        assert_eq!(&bytes[1..10], &[0xff; 9]);
        assert_eq!(bytes[10], 0x01);
    }

    #[test]
    fn model_round_trips_through_test_decoder() {
        let init = tensor_f32("w", &[2], &[1.0, -2.0]);
        let n = node(
            "Relu",
            "relu0",
            &["x".into()],
            &["y".into()],
            vec![attr_float("alpha", 0.5)],
        );
        let input = value_info("x", &[Dim::Param("batch"), Dim::Fixed(3)]);
        let output = value_info("y", &[Dim::Param("batch"), Dim::Fixed(3)]);
        let bytes = model(13, "g", &[n], &[init], &[input], &[output]);
        let top = decode(&bytes);
        assert!(
            top.iter()
                .any(|(f, w)| *f == 1 && matches!(w, Wire::Varint(8))),
            "ir_version 8"
        );
        let graph = top
            .iter()
            .find_map(|(f, w)| match (f, w) {
                (7, Wire::Len(b)) => Some(decode(b)),
                _ => None,
            })
            .unwrap();
        let node = graph
            .iter()
            .find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(b)) => Some(decode(b)),
                _ => None,
            })
            .unwrap();
        let op = node
            .iter()
            .find_map(|(f, w)| match (f, w) {
                (4, Wire::Len(b)) => Some(String::from_utf8(b.clone()).unwrap()),
                _ => None,
            })
            .unwrap();
        assert_eq!(op, "Relu");
        let init = graph
            .iter()
            .find_map(|(f, w)| match (f, w) {
                (5, Wire::Len(b)) => Some(decode(b)),
                _ => None,
            })
            .unwrap();
        let raw = init
            .iter()
            .find_map(|(f, w)| match (f, w) {
                (9, Wire::Len(b)) => Some(b.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            raw,
            [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()
        );
    }
}
