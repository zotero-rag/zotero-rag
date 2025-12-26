## Rust string conversions

Since I want to close the tab:

```
&str    -> String  | String::from(s) or s.to_string() or s.to_owned()
&str    -> &[u8]   | s.as_bytes()
&str    -> Vec<u8> | s.as_bytes().to_vec() or s.as_bytes().to_owned()
String  -> &str    | &s if possible* else s.as_str()
String  -> &[u8]   | s.as_bytes()
String  -> Vec<u8> | s.into_bytes()
&[u8]   -> &str    | s.to_vec() or s.to_owned()
&[u8]   -> String  | std::str::from_utf8(s).unwrap(), but don't**
&[u8]   -> Vec<u8> | String::from_utf8(s).unwrap(), but don't**
Vec<u8> -> &str    | &s if possible* else s.as_slice()
&Vec<u8> -> &str   | std::str::from_utf8(s).unwrap(), but don't**
Vec<u8> -> String  | std::str::from_utf8(&s).unwrap(), but don't**
Vec<u8> -> &[u8]   | String::from_utf8(s).unwrap(), but don't**

* target should have explicit type (i.e., checker can't infer that)

** handle the error properly instead
```

Source: [Stack Overflow](https://stackoverflow.com/a/41034751)
