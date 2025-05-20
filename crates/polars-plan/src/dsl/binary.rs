use super::*;
/// Specialized expressions for [`Series`] of [`DataType::String`].
pub struct BinaryNameSpace(pub(crate) Expr);

impl BinaryNameSpace {
    /// Check if a binary value contains a literal binary.
    pub fn contains_literal(self, pat: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::Contains), pat)
    }

    /// Check if a binary value ends with the given sequence.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::EndsWith), sub)
    }

    /// Check if a binary value starts with the given sequence.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0
            .map_binary(FunctionExpr::BinaryExpr(BinaryFunction::StartsWith), sub)
    }

    /// Return the size (number of bytes) in each element.
    pub fn size_bytes(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Size))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_decode(self, strict: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::HexDecode(strict)))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_encode(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::HexEncode))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_decode(self, strict: bool) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Base64Decode(
                strict,
            )))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_encode(self) -> Expr {
        self.0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::Base64Encode))
    }

    #[cfg(feature = "binary_encoding")]
    pub fn from_buffer(self, to_type: DataType, is_little_endian: bool) -> Expr {
        let leaf_type = to_type.leaf_dtype();
        let shape = to_type.get_shape();

        let call_to_type = if let Some(ref shape) = shape {
            DataType::Array(
                Box::new(leaf_type.clone()),
                shape.iter().product(),
            )
        } else {
            to_type
        };

        let result = self
            .0
            .map_unary(FunctionExpr::BinaryExpr(BinaryFunction::FromBuffer(
                call_to_type,
                is_little_endian,
            )));

        if let Some(shape) = shape {
            let mut dimensions: Vec<ReshapeDimension> = shape
                .iter()
                .map(|&v| ReshapeDimension::new(v as i64))
                .collect();
            dimensions.insert(0, ReshapeDimension::Infer);

            result.map_unary(FunctionExpr::Reshape(dimensions))
        } else {
            result
        }
    }
}
