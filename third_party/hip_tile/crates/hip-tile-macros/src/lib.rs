//! `#[kernel]`: one source function, a device entry point plus a typed host launcher.
//!
//! ```ignore
//! #[kernel(lds = lds_tile_bytes::<f32>(64, 16))]
//! fn saxpy(a: f32, x: Tensor<f32>, y: TensorMut<f32>) { /* device code */ }
//! ```
//!
//! On `target_arch = "amdgpu"` this becomes an `extern "gpu-kernel"` symbol named
//! `saxpy`. On the host it becomes `mod saxpy` with `NAME`, `BLOCK_THREADS`,
//! `LDS_BYTES` and `launch(module, stream, grid, a, x, y)`; the body is not compiled.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{Error, Expr, FnArg, Ident, ItemFn, Pat, Result, Token, parse_macro_input};

struct Args {
    lds: Option<Expr>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut lds = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key != "lds" {
                return Err(Error::new(key.span(), "unknown option, expected `lds = <expr>`"));
            }
            if lds.is_some() {
                return Err(Error::new(key.span(), "duplicate `lds` option"));
            }
            input.parse::<Token![=]>()?;
            lds = Some(input.parse()?);
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(Self { lds })
    }
}

/// Mark a plain `fn` as a GPU kernel. Optional `lds = <const expr>` is the dynamic
/// LDS size in bytes (a `u32` const expression evaluated in the caller's scope).
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as Args);
    let func = parse_macro_input!(item as ItemFn);
    expand(args, func).unwrap_or_else(Error::into_compile_error).into()
}

fn expand(args: Args, func: ItemFn) -> Result<TokenStream2> {
    let sig = &func.sig;
    if let Some(tok) = sig.asyncness {
        return Err(Error::new(tok.span(), "kernels cannot be async"));
    }
    if let Some(tok) = sig.constness {
        return Err(Error::new(tok.span(), "kernels cannot be const"));
    }
    if let Some(tok) = sig.unsafety {
        return Err(Error::new(tok.span(), "kernels are declared as safe `fn`; the macro adds `unsafe`"));
    }
    if let Some(abi) = &sig.abi {
        return Err(Error::new(abi.span(), "the macro sets the ABI; remove it"));
    }
    if !sig.generics.params.is_empty() {
        return Err(Error::new(sig.generics.span(), "kernels cannot be generic"));
    }
    if !matches!(sig.output, syn::ReturnType::Default) {
        return Err(Error::new(sig.output.span(), "kernels cannot return a value"));
    }

    let mut names = Vec::new();
    let mut types = Vec::new();
    for input in &sig.inputs {
        let FnArg::Typed(arg) = input else {
            return Err(Error::new(input.span(), "kernels cannot take `self`"));
        };
        let Pat::Ident(pat) = &*arg.pat else {
            return Err(Error::new(arg.pat.span(), "kernel parameters must be plain identifiers"));
        };
        if pat.subpat.is_some() || pat.by_ref.is_some() {
            return Err(Error::new(arg.pat.span(), "kernel parameters must be plain identifiers"));
        }
        names.push(pat.ident.clone());
        types.push((*arg.ty).clone());
    }

    let name = &sig.ident;
    let name_str = name.to_string();
    let params = &sig.inputs;
    let attrs = &func.attrs;
    let body = &func.block;
    let lds = args.lds.map_or_else(|| quote!(0), |e| quote!(#e));
    let n = names.len();
    let locals: Vec<Ident> = names.iter().map(|i| format_ident!("__arg_{}", i)).collect();

    Ok(quote! {
        const _: () = {
            fn assert_kernel_arg<T: ::hip_tile::KernelArg>() {}
            #[allow(dead_code)]
            fn check_params() { #(assert_kernel_arg::<#types>();)* }
        };

        #[cfg(target_arch = "amdgpu")]
        #(#attrs)*
        #[unsafe(no_mangle)]
        #[allow(unused_unsafe, clippy::missing_safety_doc)]
        pub unsafe extern "gpu-kernel" fn #name(#params) {
            unsafe #body
        }

        #[cfg(not(target_arch = "amdgpu"))]
        #[allow(unused_imports)]
        pub mod #name {
            use super::*;

            pub const NAME: &str = #name_str;
            pub const BLOCK_THREADS: u32 = ::hip_tile::config::BLOCK_THREADS;
            pub const LDS_BYTES: u32 = #lds;

            /// Launch on `stream` (default stream when `None`) with `grid` blocks of
            /// `BLOCK_THREADS` threads. Asynchronous: synchronise before reading results.
            #[allow(clippy::too_many_arguments)]
            pub fn launch(
                __module: &::hip_rt::Module,
                __stream: ::core::option::Option<&::hip_rt::Stream>,
                __grid: impl ::core::convert::Into<::hip_rt::Dim3>,
                #(#names: #types),*
            ) -> ::hip_rt::Result<()> {
                let __func = __module.function(NAME)?;
                #(let mut #locals = #names;)*
                let mut __args: [*mut ::core::ffi::c_void; #n] =
                    [#(&mut #locals as *mut _ as *mut ::core::ffi::c_void),*];
                // SAFETY: one pointer per parameter, typed as the kernel's parameters
                // (checked `KernelArg`s); the device pointers are the caller's contract.
                unsafe {
                    __func.launch(
                        __stream,
                        __grid.into(),
                        BLOCK_THREADS.into(),
                        LDS_BYTES,
                        &mut __args,
                    )
                }
            }
        }
    })
}
